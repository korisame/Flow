import Foundation
import NaturalLanguage
import FluidAudio

// Reimplemented helpers: the original complete files were not recoverable.
public enum FlowLanguage {
    public static func detect(_ text: String) -> String? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage?.rawValue
    }
}
public enum WavIO {
    public static func loadAsFlowSamples(_ url: URL) throws -> [Float] {
        try AudioConverter().resampleAudioFile(url)
    }
}

// Preserve ambiguous speech, including legitimate thanks/goodbyes. The old
// stock phrase list deleted these without audio evidence. Only explicit
// non-speech markers are removed in this reconstruction.
public func stripStockPhrases(_ text: String) -> String {
    regexReplace(text, pattern: "(?i)\\[(musica|applausi|risate|music|applause)\\]", with: "")
}
public func stripTrailingRepeat(_ text: String) -> String { stripRepeatLoop(text) }
public func stripTrailingEcho(_ text: String) -> String {
    // A repeated sentence may be intentional; do not drop it heuristically.
    text
}
public func isHallucination(_ text: String, expectedLanguage: String?) -> Bool {
    let compact = text.filter { !$0.isWhitespace }
    return compact.count >= 20 && Set(compact).count <= 2
}
