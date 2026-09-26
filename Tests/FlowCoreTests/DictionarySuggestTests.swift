import XCTest
@testable import FlowCore

final class DictionarySuggestTests: XCTestCase {
    func testProposesRecurringMidSentenceNames() {
        let transcripts = [
            "Domani vado da Riminiwood a prendere i mobili.",
            "Ho scritto a Riminiwood per il preventivo.",
            "Chiedi a Riminiwood se ha ancora quel comodino.",
        ]
        let out = suggestDictionaryTerms(from: transcripts, existing: [])
        XCTAssertEqual(out.first?.term, "Riminiwood")
        XCTAssertEqual(out.first?.count, 3)
    }

    func testIgnoresSentenceInitialCapitals() {
        let transcripts = Array(repeating: "Allora facciamo così. Guarda che poi ti dico.", count: 6)
        let terms = suggestDictionaryTerms(from: transcripts, existing: []).map(\.term)
        XCTAssertFalse(terms.contains("Allora"))
        XCTAssertFalse(terms.contains("Guarda"))
    }

    func testIgnoresTermsAlreadyKnown() {
        let transcripts = Array(repeating: "Ok, scrivi a Pontevecchio del preventivo.", count: 5)
        XCTAssertFalse(suggestDictionaryTerms(from: transcripts, existing: []).isEmpty)
        XCTAssertTrue(suggestDictionaryTerms(from: transcripts, existing: ["Pontevecchio"]).isEmpty)
        // Terms covered by a built-in alias count as known too.
        let known = Array(repeating: "Ok, scrivi a Graziella del preventivo.", count: 5)
        XCTAssertTrue(suggestDictionaryTerms(from: known, existing: []).isEmpty)
    }

    func testCountsEachTermOncePerDictation() {
        // One transcript repeating a word must not reach the threshold on its own.
        let spammy = "Ok, parlo di Nadia e Nadia e ancora Nadia e Nadia."
        XCTAssertTrue(suggestDictionaryTerms(from: [spammy], existing: []).isEmpty)
    }

    func testSkipsNumbersAcronymsAndShortWords() {
        let transcripts = Array(repeating: "Scrivi il codice GBLT180 e la sigla IVA per il Bar.", count: 4)
        let terms = suggestDictionaryTerms(from: transcripts, existing: []).map(\.term)
        XCTAssertFalse(terms.contains("GBLT180"))
        XCTAssertFalse(terms.contains("IVA"))
        XCTAssertFalse(terms.contains("Bar"))  // under the 4-character floor
    }
}

