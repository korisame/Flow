import Foundation
import FluidAudio

/// ASR engine: Parakeet Ultra (CoreML via FluidAudio).
public final class Transcriber {
    public static let modelRepository = "FluidInference/parakeet-ultra-coreml"
    public static let modelRevision = "95eaa59a39d4394f047a4dc5cce480388a60d1b6"
    public static let selectedModelLabel = "\(modelRepository)@\(modelRevision)"

    private var asr: AsrManager?
    private var decoderLayers: Int = 2
    public let modelLabel = Transcriber.selectedModelLabel

    public init() {}

    private let loadLock = NSLock()
    /// The load currently in flight, if any. Without this, two callers that
    /// both observe `asr == nil` (typically preloadASR on Fn press and the
    /// worker's ensureLoaded when the recording ends first) each run a full
    /// CoreML/ANE materialisation of the 460 MB model. Measured: that turns a
    /// ~1s reload into a 11-35s stall with the HUD frozen.
    private var loadTask: Task<Void, Error>? = nil
    /// Bumped by unload() so a load that completes afterwards cannot resurrect
    /// a model the user (or the idle timer) explicitly asked to free.
    private var loadGen: UInt64 = 0

    public var isLoaded: Bool { loadLock.lock(); defer { loadLock.unlock() }; return asr != nil }

    /// Download (first run) and load the pinned Ultra CoreML models. Concurrent callers
    /// share a single in-flight load instead of starting duplicates.
    public func load() async throws {
        loadLock.lock()
        if asr != nil { loadLock.unlock(); return }
        if let inFlight = loadTask { loadLock.unlock(); return try await inFlight.value }
        loadGen &+= 1
        let gen = loadGen
        let task = Task<Void, Error>.detached { [self] in
            var loaded: AsrManager? = nil
            // Release the in-flight slot on EVERY exit, success or throw.
            // Leaving a failed task pinned here would make every later load()
            // await it and re-throw the same cached error forever.
            defer {
                loadLock.lock()
                if gen == loadGen { self.loadTask = nil }
                let stale = (gen != loadGen)
                if let mgr = loaded, !stale { self.asr = mgr }
                loadLock.unlock()
                // An unload landed while this load was running: drop the model
                // we just built instead of silently re-holding ~2.7 GB.
                if stale, let mgr = loaded { Task.detached { await mgr.cleanup() } }
            }
            var revisionOverrides = ModelRegistry.revisionOverrides
            revisionOverrides[Self.modelRepository] = Self.modelRevision
            ModelRegistry.revisionOverrides = revisionOverrides
            let models = try await AsrModels.downloadAndLoad(version: .ultra)
            let mgr = AsrManager(config: .default)
            try await mgr.loadModels(models)
            let layers = await mgr.decoderLayerCount
            loadLock.lock(); self.decoderLayers = layers; loadLock.unlock()
            loaded = mgr
        }
        loadTask = task
        loadLock.unlock()
        try await task.value
    }

    /// Load the model if it was idle-unloaded. Cheap when already loaded.
    public func ensureLoaded() async throws {
        try await load()
    }

    public struct Result {
        public let text: String
        public let transcribeS: Double
        /// Seconds spent waiting for the model to be resident before inference
        /// could start. Non-zero only after an idle unload. Reported separately
        /// so a cold reload stops showing up as unattributed "total" time.
        public let loadS: Double
    }

    /// Transcribe 16 kHz mono float32 samples. `language` is an optional
    /// ISO code hint ("it", "en", ...); nil = auto/multilingual.
    public func transcribe(_ samples: [Float], language: String? = nil) async throws -> Result {
        let tLoad = Date()
        try await ensureLoaded()
        let loadS = -tLoad.timeIntervalSinceNow
        if loadS > 0.5 {
            flowLog(String(format: "[asr] model reload took %.2fs (idle unload)", loadS))
        }
        loadLock.lock(); let mgr = asr; let layers = decoderLayers; loadLock.unlock()
        guard let mgr else {
            throw NSError(domain: "Flow", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "ASR model not loaded"])
        }
        let lang = language.flatMap { Language(rawValue: $0.lowercased()) }
        var state = TdtDecoderState.make(decoderLayers: layers)
        let t0 = Date()
        let result = try await mgr.transcribe(samples, decoderState: &state, language: lang)
        return Result(text: result.text, transcribeS: -t0.timeIntervalSinceNow, loadS: loadS)
    }

    /// Release the CoreML models and their ANE/GPU memory (the ~2.7 GB
    /// IOAccelerator footprint). Reloading is NOT free: measured 0.02s at best
    /// but 15-47s in the bad cases, paid on the next dictation. Unload only
    /// after a long idle — see asrIdleUnloadSeconds.
    public func unload() async {
        loadLock.lock()
        let mgr = asr
        asr = nil
        // Invalidate any load still in flight so it cannot re-populate `asr`
        // right after we freed it.
        loadGen &+= 1
        loadTask = nil
        loadLock.unlock()
        if let mgr { await mgr.cleanup() }
    }
}
