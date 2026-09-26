import Foundation

public struct DictionarySuggestion: Equatable {
    public let term: String
    public let count: Int
}

/// Words that look capitalised in a transcript for structural reasons rather
/// than because they are names: sentence starts contribute most of these.
private let SUGGEST_STOPWORDS: Set<String> = [
    "allora", "quindi", "però", "perché", "poi", "ecco", "certo", "senti",
    "guarda", "dimmi", "fammi", "fai", "puoi", "devi", "vorrei", "adesso",
    "oggi", "domani", "ieri", "questo", "questa", "quello", "quella", "come",
    "cosa", "dove", "quando", "anche", "ancora", "sempre", "invece", "magari",
    "praticamente", "sostanzialmente", "comunque", "insomma", "diciamo",
    "the", "this", "that", "then", "there", "here", "what", "when", "where",
    "okay", "yeah", "well", "just", "very", "with", "from", "have", "hello",
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

/// Propose dictionary candidates from the user's own dictation history.
///
/// A term qualifies when it is capitalised mid-sentence (so the capital is a
/// property of the word, not of the position), appears repeatedly across
/// different dictations, and is not already known. This surfaces the names and
/// jargon the user actually says instead of asking them to remember them.
public func suggestDictionaryTerms(from transcripts: [String],
                                   existing: [String],
                                   minOccurrences: Int = 3,
                                   limit: Int = 25) -> [DictionarySuggestion] {
    var known = Set(existing.map { $0.lowercased() })
    for (alias, canonical) in DICT_ALIASES {
        known.insert(alias.lowercased())
        known.insert(canonical.lowercased())
    }

    var counts: [String: Int] = [:]
    var display: [String: String] = [:]

    for transcript in transcripts {
        // Split into sentences so the first word of each can be ignored: its
        // capital says nothing about the word itself.
        let sentences = transcript.split(whereSeparator: { ".!?\n\u{2026}".contains($0) })
        var seenInThis = Set<String>()
        for sentence in sentences {
            let words = sentence.split(whereSeparator: { $0 == " " || $0 == "," || $0 == ";" || $0 == ":" })
            for (index, rawWord) in words.enumerated() {
                if index == 0 { continue }
                let word = rawWord.trimmingCharacters(
                    in: CharacterSet(charactersIn: "\"'()[]«»“”‘’-–—"))
                guard word.count >= 4, word.count <= 24 else { continue }
                guard let first = word.first, first.isUppercase else { continue }
                // Letters only: no numbers, no code fragments, no URLs.
                guard word.allSatisfy({ $0.isLetter || $0 == "'" }) else { continue }
                // All-caps runs are usually acronyms the ASR spelled out.
                guard word.dropFirst().contains(where: { $0.isLowercase }) else { continue }
                let key = word.lowercased()
                guard !known.contains(key), !SUGGEST_STOPWORDS.contains(key) else { continue }
                // Count each term once per dictation so a single repetitive
                // transcript cannot promote its own vocabulary.
                guard !seenInThis.contains(key) else { continue }
                seenInThis.insert(key)
                counts[key, default: 0] += 1
                if display[key] == nil { display[key] = word }
            }
        }
    }

    return counts
        .filter { $0.value >= minOccurrences }
        .sorted { ($0.value, $1.key) > ($1.value, $0.key) }
        .prefix(limit)
        .compactMap { key, count in
            display[key].map { DictionarySuggestion(term: $0, count: count) }
        }
}

