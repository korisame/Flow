import Foundation
import AVFoundation

public enum FlowAudio {
    public static let sampleRate = 16_000
    public static let blockMs = 80
    /// Keep capturing this long after Fn release so in-flight blocks land
    /// in the buffer (otherwise the end of the dictation gets cropped).
    public static let tailCaptureS = 0.25
    /// RMS threshold shared by hands-free segmentation and the batch gate.
    /// It is deliberately above the trailing-silence threshold so low-level
    /// room/mic noise cannot be mistaken for a dictation.
    public static let speechRMS: Float = 0.018
    public static let trimRMS: Float = 0.005
    public static let vadWindowMs = 30
    public static let minVoicedWindows = 3
    public static let minConsecutiveVoicedWindows = 2
    public static let silenceSec = 1.5
    public static let minUttSec = 0.4
}

/// Energy-only voice-activity result used to keep silent captures out of ASR.
/// A single click can exceed an RMS threshold, so valid speech needs both a
/// short contiguous run and enough voiced windows overall.
public struct FlowVoiceActivity: Equatable, Sendable {
    public let totalWindows: Int
    public let voicedWindows: Int
    public let longestVoicedRun: Int
    public let peakRMS: Float

    public var hasVoice: Bool {
        voicedWindows >= FlowAudio.minVoicedWindows &&
        longestVoicedRun >= FlowAudio.minConsecutiveVoicedWindows
    }
}

/// Inspect 16 kHz mono samples with the same RMS threshold used by the
/// recorder's legacy hands-free VAD. This is intentionally independent of
/// recorder mode so push-to-talk is gated too.
public func voiceActivity(in audio: [Float]) -> FlowVoiceActivity {
    guard !audio.isEmpty else {
        return FlowVoiceActivity(totalWindows: 0, voicedWindows: 0,
                                 longestVoicedRun: 0, peakRMS: 0)
    }

    let window = max(1, Int(Double(FlowAudio.sampleRate) * Double(FlowAudio.vadWindowMs) / 1000.0))
    var totalWindows = 0
    var voicedWindows = 0
    var currentVoicedRun = 0
    var longestVoicedRun = 0
    var peakRMS: Float = 0
    var start = 0

    while start < audio.count {
        let end = min(start + window, audio.count)
        var sum: Float = 0
        for index in start..<end {
            let sample = audio[index]
            sum += sample * sample
        }
        let rms = sqrt(sum / Float(end - start))
        peakRMS = max(peakRMS, rms)
        totalWindows += 1
        if rms >= FlowAudio.speechRMS {
            voicedWindows += 1
            currentVoicedRun += 1
            longestVoicedRun = max(longestVoicedRun, currentVoicedRun)
        } else {
            currentVoicedRun = 0
        }
        start = end
    }

    return FlowVoiceActivity(totalWindows: totalWindows,
                             voicedWindows: voicedWindows,
                             longestVoicedRun: longestVoicedRun,
                             peakRMS: peakRMS)
}

/// Microphone recorder: 16 kHz mono float32 with hands-free RMS VAD.
/// Port of flow.py Recorder (sounddevice -> AVAudioEngine).
public final class Recorder {
    private var engine: AVAudioEngine?
    private var converter: AVAudioConverter?
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: 16_000, channels: 1,
                                          interleaved: false)!

    private var buf: [[Float]] = []
    private var sessionBuf: [[Float]] = []
    private let bufLock = NSLock()

    private var recording = false
    private var handsFree = false
    private var onUtterance: (([Float]) -> Void)?

    private var lastSpeechT: Double? = nil
    private var hadSpeech = false
    private var lastCbT: Double = 0

    private var watchdog: Thread?
    private var watchdogRun = false

    private let minSamples = Int(Double(FlowAudio.sampleRate) * FlowAudio.minUttSec)
    private let overlapSamples = Int(Double(FlowAudio.sampleRate) * 0.25)

    /// Livello del microfono smussato, 0...1, per la forma d'onda dell'HUD.
    private var levelValue: Float = 0
    private let levelLock = NSLock()

    public init() {}

    public var isRecording: Bool { recording }
    public var isHandsFree: Bool { handsFree }

    /// Ultimo livello udito (RMS normalizzato con attacco veloce e rilascio lento).
    public var inputLevel: Float {
        levelLock.lock(); defer { levelLock.unlock() }
        return levelValue
    }

    private func updateLevel(_ rms: Float) {
        // La voce parlata sta intorno a 0.02...0.25 di RMS: normalizzo li'.
        let normalized = min(1, max(0, (rms - 0.004) / 0.16))
        levelLock.lock()
        levelValue = normalized > levelValue
            ? levelValue + (normalized - levelValue) * 0.6      // attacco
            : levelValue + (normalized - levelValue) * 0.18     // rilascio
        levelLock.unlock()
    }

    public func start(handsFree: Bool = false, onUtterance: (([Float]) -> Void)? = nil) throws {
        bufLock.lock()
        buf = []
        sessionBuf = []
        bufLock.unlock()
        self.handsFree = handsFree
        self.onUtterance = onUtterance
        lastSpeechT = nil
        hadSpeech = false
        lastCbT = ProcessInfo.processInfo.systemUptime

        // Flag FIRST, engine second. openEngine() blocks for tens of ms (up to
        // ~0.8s on a retry); if `recording` only flipped afterwards, an Fn
        // release arriving in that window would see isRecording == false, skip
        // the stop, and leave a recording nobody ever ends — the next press
        // then wiped the buffer and the dictation was lost with no paste.
        recording = true
        do {
            try openEngine()
        } catch {
            recording = false
            self.handsFree = false
            self.onUtterance = nil
            throw error
        }
        // A stop that arrived from another path (IPC, quit) while the engine was
        // opening already cleared the flag: don't leave a live engine behind
        // with no watchdog and no owner.
        guard recording else {
            teardownEngine()
            return
        }
        startWatchdog()
    }

    /// Promote a running push-to-talk stream to hands-free in place
    /// (double-press) without recreating the stream. Hands-free records the
    /// whole session continuously and transcribes ONCE on stop (no live VAD
    /// segmentation): batch transcription is higher quality and Parakeet is
    /// fast enough that the wait is short.
    public func promoteToHandsFree() {
        self.handsFree = true
        self.onUtterance = nil
    }

    private func openEngine() throws {
        var lastError: Error? = nil
        for attempt in 0..<2 {
            do {
                let e = AVAudioEngine()
                let input = e.inputNode
                let inFormat = input.inputFormat(forBus: 0)
                guard inFormat.sampleRate > 0 else {
                    throw NSError(domain: "Flow", code: 10,
                                  userInfo: [NSLocalizedDescriptionKey: "no input device"])
                }
                let conv = AVAudioConverter(from: inFormat, to: outFormat)
                let frames = AVAudioFrameCount(inFormat.sampleRate * Double(FlowAudio.blockMs) / 1000.0)
                input.installTap(onBus: 0, bufferSize: frames, format: inFormat) { [weak self] pcm, _ in
                    self?.audioCallback(pcm)
                }
                e.prepare()
                try e.start()
                // Drop any previous engine before taking ownership of the new
                // one, otherwise a re-entrant start leaks an AVAudioEngine whose
                // input tap keeps feeding audioCallback into the same buffers.
                teardownEngine()
                self.engine = e
                self.converter = conv
                return
            } catch {
                lastError = error
                flowLog("[rec] engine open attempt \(attempt + 1) failed: \(error)")
                Thread.sleep(forTimeInterval: 0.4)
            }
        }
        throw lastError ?? NSError(domain: "Flow", code: 11)
    }

    private func audioCallback(_ pcm: AVAudioPCMBuffer) {
        // Convert to 16 kHz mono float32. Never throw out of the callback.
        guard let conv = converter else { return }
        let ratio = 16_000.0 / pcm.format.sampleRate
        let cap = AVAudioFrameCount(Double(pcm.frameLength) * ratio + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: max(cap, 64)) else { return }
        var fed = false
        var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return pcm
        }
        guard err == nil, let ch = out.floatChannelData, out.frameLength > 0 else { return }
        let chunk = Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))

        bufLock.lock()
        buf.append(chunk)
        sessionBuf.append(chunk)
        bufLock.unlock()
        lastCbT = ProcessInfo.processInfo.systemUptime

        var sum: Float = 0
        for s in chunk { sum += s * s }
        let rms = sqrt(sum / Float(max(1, chunk.count)))
        updateLevel(rms)

        // Hands-free RMS VAD segmentation
        guard handsFree, let onUtterance else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if rms > FlowAudio.speechRMS {
            lastSpeechT = now
            hadSpeech = true
        }
        if hadSpeech, let lastSpeech = lastSpeechT, now - lastSpeech > FlowAudio.silenceSec {
            bufLock.lock()
            let audio = buf.flatMap { $0 }
            if audio.count > overlapSamples {
                buf = [Array(audio.suffix(overlapSamples))]
            } else {
                buf = [audio]
            }
            bufLock.unlock()
            hadSpeech = false
            lastSpeechT = nil
            if audio.count >= minSamples {
                let payload = audio
                Thread.detachNewThread { onUtterance(payload) }
            }
        }
    }

    /// Stop and return the recorded 16 kHz mono samples (with tail capture).
    public func stop() -> [Float] {
        // Idempotent: a duplicate stop must not return an empty buffer and
        // make the caller believe the dictation was silent.
        guard recording else { return [] }
        recording = false
        // Tail capture: the last word is often still in flight. audioCallback
        // keeps appending to `buf` regardless of the flag, so the tail is still
        // collected while we wait here.
        Thread.sleep(forTimeInterval: FlowAudio.tailCaptureS)
        handsFree = false
        onUtterance = nil
        levelLock.lock(); levelValue = 0; levelLock.unlock()
        stopWatchdog()
        teardownEngine()
        bufLock.lock()
        let audio = buf.flatMap { $0 }
        buf = []
        bufLock.unlock()
        return audio
    }

    /// Snapshot of the whole session without clearing.
    public func sessionAudio() -> [Float] {
        bufLock.lock()
        defer { bufLock.unlock() }
        return sessionBuf.flatMap { $0 }
    }

    private func teardownEngine() {
        if let e = engine {
            e.inputNode.removeTap(onBus: 0)
            e.stop()
        }
        engine = nil
        converter = nil
    }

    // MARK: - Watchdog (port: restart stream when callbacks stall > 1.2 s)

    private func startWatchdog() {
        watchdogRun = true
        let t = Thread { [weak self] in
            while let self, self.watchdogRun {
                Thread.sleep(forTimeInterval: 0.3)
                guard self.recording else { continue }
                let now = ProcessInfo.processInfo.systemUptime
                if now - self.lastCbT > 1.2 {
                    flowLog("[rec] watchdog: input stalled, restarting stream")
                    self.teardownEngine()
                    do {
                        try self.openEngine()
                        self.lastCbT = ProcessInfo.processInfo.systemUptime
                    } catch {
                        flowLog("[rec] watchdog restart failed: \(error)")
                    }
                }
            }
        }
        t.name = "flow-rec-watchdog"
        t.start()
        watchdog = t
    }

    private func stopWatchdog() {
        watchdogRun = false
        watchdog = nil
    }
}

/// RMS-based trailing-silence trim (port of the numpy fallback:
/// rms_threshold=0.005, win_ms=30, keep_pad_ms=200). An entirely silent
/// capture is represented as an empty buffer, not as the original samples.
public func trimTrailingSilence(_ audio: [Float]) -> [Float] {
    if audio.isEmpty { return audio }
    let win = Int(Double(FlowAudio.sampleRate) * Double(FlowAudio.vadWindowMs) / 1000.0)
    let pad = Int(Double(FlowAudio.sampleRate) * 200.0 / 1000.0)
    var lastSpeechEnd = 0
    var i = 0
    while i < audio.count {
        let end = min(i + win, audio.count)
        var sum: Float = 0
        for k in i..<end { sum += audio[k] * audio[k] }
        let rms = sqrt(sum / Float(end - i))
        if rms > FlowAudio.trimRMS { lastSpeechEnd = end }
        i = end
    }
    if lastSpeechEnd == 0 { return [] }
    let cut = min(audio.count, lastSpeechEnd + pad)
    return Array(audio[0..<cut])
}

