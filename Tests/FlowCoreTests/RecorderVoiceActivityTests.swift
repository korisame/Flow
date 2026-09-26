import XCTest
@testable import FlowCore

final class RecorderVoiceActivityTests: XCTestCase {
    func testTrimReturnsEmptyForEntirelySilentCapture() {
        let audio = samples(amplitude: 0.004, milliseconds: 1_200)

        XCTAssertTrue(trimTrailingSilence(audio).isEmpty)
        XCTAssertTrue(trimTrailingSilence(Array(repeating: 0, count: 17)).isEmpty)
    }

    func testVoiceActivityRejectsLowLevelRoomNoise() {
        let ambientNoise = samples(amplitude: 0.010, milliseconds: 1_200)
        let trimmed = trimTrailingSilence(ambientNoise)
        let activity = voiceActivity(in: trimmed)

        XCTAssertFalse(trimmed.isEmpty, "The VAD, not only trailing trim, rejects ambient noise.")
        XCTAssertFalse(activity.hasVoice)
        XCTAssertEqual(activity.voicedWindows, 0)
    }

    func testVoiceActivityRejectsSingleLoudClick() {
        var audio = samples(amplitude: 0, milliseconds: 1_200)
        for index in 800..<960 { audio[index] = 0.08 }

        let activity = voiceActivity(in: audio)

        XCTAssertFalse(activity.hasVoice)
        XCTAssertEqual(activity.voicedWindows, 1)
        XCTAssertEqual(activity.longestVoicedRun, 1)
    }

    func testVoiceActivityAcceptsSustainedSpeechEnergy() {
        let audio = samples(amplitude: 0, milliseconds: 300)
            + samples(amplitude: 0.030, milliseconds: 180)
            + samples(amplitude: 0, milliseconds: 300)
        let activity = voiceActivity(in: trimTrailingSilence(audio))

        XCTAssertTrue(activity.hasVoice)
        XCTAssertGreaterThanOrEqual(activity.voicedWindows, FlowAudio.minVoicedWindows)
        XCTAssertGreaterThanOrEqual(activity.longestVoicedRun, FlowAudio.minConsecutiveVoicedWindows)
    }

    private func samples(amplitude: Float, milliseconds: Int) -> [Float] {
        Array(repeating: amplitude,
              count: FlowAudio.sampleRate * milliseconds / 1_000)
    }
}

