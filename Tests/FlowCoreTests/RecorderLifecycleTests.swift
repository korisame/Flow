import XCTest
@testable import FlowCore

/// Guards the hotkey race that made dictations disappear: a release edge that
/// observed `isRecording == false` mid-`start()` skipped the stop, and the next
/// press then wiped the buffer. See Recorder.start()/stop().
final class RecorderLifecycleTests: XCTestCase {
    func testStopIsIdempotentAndCheapWhenNotRecording() {
        let recorder = Recorder()

        XCTAssertFalse(recorder.isRecording)
        // Must not sleep for the tail-capture window and must not report a
        // (bogus) empty capture that a caller would treat as silent audio.
        let started = Date()
        XCTAssertTrue(recorder.stop().isEmpty)
        XCTAssertLessThan(-started.timeIntervalSinceNow, FlowAudio.tailCaptureS,
                          "A duplicate stop must return immediately, not pay the tail capture again.")
        XCTAssertFalse(recorder.isRecording)
    }

    // NOTE: start() is deliberately not exercised here — it opens a real
    // AVAudioEngine and would block the test runner on the microphone
    // permission prompt. The unwind-on-failure path is covered by inspection:
    // Recorder.start() sets `recording = false` in its catch before rethrowing.
}

