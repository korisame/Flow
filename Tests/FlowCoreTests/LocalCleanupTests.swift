import XCTest
@testable import FlowCore

final class LocalCleanupTests: XCTestCase {
    func testMissingPromptFailsSafely() {
        XCTAssertNil(LocalCleanupModel().generate(messages: [], maxTokens: 12))
    }
    func testOversizedPromptFailsWithoutLaunching() {
        XCTAssertNil(LocalCleanupModel().generate(messages: [("user", String(repeating: "x", count: 16001))], maxTokens: 12))
    }
    func testMissingRuntimePreservesText() {
        setenv("FLOW_PUNCTUATION_WORKER", "/nonexistent/flow-worker.py", 1)
        defer { unsetenv("FLOW_PUNCTUATION_WORKER") }
        let model = LocalCleanupModel()
        XCTAssertNil(model.generate(messages: [("user", "ciao come stai")], maxTokens: 12))
        model.unload()
    }
    func testConservativeFiltersKeepLegitimateSpeech() {
        XCTAssertEqual(stripStockPhrases("grazie per tutto"), "grazie per tutto")
        XCTAssertEqual(stripTrailingEcho("Vieni qui. Vieni qui."), "Vieni qui. Vieni qui.")
        XCTAssertFalse(isHallucination("Il cliente ha detto grazie", expectedLanguage: "it"))
        XCTAssertTrue(isHallucination(String(repeating: "a", count: 25), expectedLanguage: nil))
    }
    func testLanguageDetection() {
        XCTAssertEqual(FlowLanguage.detect("Buongiorno, vorrei sapere quando arriveranno i documenti richiesti."), "it")
    }
}
