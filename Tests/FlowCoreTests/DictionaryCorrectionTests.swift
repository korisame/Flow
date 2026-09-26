import XCTest
@testable import FlowCore

/// Cases taken from the real dictation history in ~/.flow/flow.sqlite, so a
/// regression here is a regression Shaun would actually hit.
final class DictionaryCorrectionTests: XCTestCase {
    private let terms = [
        "Graziella", "Braccialini", "Carrelli", "Beeper", "Hermes", "Subito",
        "Codex", "ChatGPT", "showroom", "gioielleria", "Sara Nocentini",
        "Porsche", "eToro", "iPhone", "Python", "Flow", "AIR",
    ]

    private func fix(_ s: String) -> String {
        correctDictionaryTerms(s, terms: terms)
    }

    func testFuzzyFixesRealAsrMisspellings() {
        XCTAssertEqual(fix("mandalo a Graciella"), "mandalo a Graziella")
        XCTAssertEqual(fix("scrivi su beper."), "scrivi su Beeper.")
        XCTAssertEqual(fix("chiedi a chatGPD"), "chiedi a ChatGPT")
        XCTAssertEqual(fix("mandalo a Sarrano Centini"), "mandalo a Sara Nocentini")
        XCTAssertEqual(fix("mandalo a saranno centini"), "mandalo a Sara Nocentini")
    }

    func testDoesNotRewriteOrdinaryItalianWords() {
        // "carrello" (shopping cart) is one edit from the name "Carrelli" and
        // was rewritten 8 times across the real history before it was blocked.
        XCTAssertEqual(fix("aggiungi al carrello"), "aggiungi al carrello")
        XCTAssertEqual(fix("svuota il carrello."), "svuota il carrello.")
        XCTAssertEqual(fix("ordinare carrelli omologabili"), "ordinare carrelli omologabili")
        // "subito" the adverb must not become the marketplace.
        XCTAssertEqual(fix("fallo subito per favore"), "fallo subito per favore")
        // ...but the marketplace, written as such, still canonicalises.
        XCTAssertEqual(fix("l'ho messo su Subito"), "l'ho messo su Subito")
        // A jeweller is not a jewellery shop.
        XCTAssertEqual(fix("ho parlato col gioielliere"), "ho parlato col gioielliere")
    }

    func testKeepsSentenceInitialCapitalOnLowercaseTerms() {
        XCTAssertEqual(fix("Showroom aperto domani."), "Showroom aperto domani.")
    }

    func testExactAndAliasCorrectionsStillWork() {
        XCTAssertEqual(fix("vado in shoom."), "vado in showroom.")
        XCTAssertEqual(fix("la toielleria è chiusa"), "la gioielleria è chiusa")
        XCTAssertEqual(fix("parliamo di codex oggi"), "parliamo di Codex oggi")
    }
}

