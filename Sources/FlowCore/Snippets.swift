import Foundation

/// Port of flow.py _apply_snippets.
public func applySnippets(_ text: String, store: HistoryDB, enabled: Bool) -> (String, Int) {
    if text.isEmpty || !enabled { return (text, 0) }
    let snippets = store.listSnippets()
    if snippets.isEmpty { return (text, 0) }
    let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)

    var lowerMap: [String: HistoryDB.Snippet] = [:]
    for s in snippets {
        lowerMap[s.phrase.trimmingCharacters(in: .whitespaces).lowercased()] = s
    }

    var key = stripped.lowercased()
    if let re = try? NSRegularExpression(pattern: "^(?:snippet|inserisci snippet|espandi snippet)\\s+(.+)$",
                                         options: [.caseInsensitive]),
       let m = re.firstMatch(in: stripped, range: NSRange(location: 0, length: (stripped as NSString).length)),
       m.numberOfRanges > 1 {
        key = (stripped as NSString).substring(with: m.range(at: 1))
            .trimmingCharacters(in: .whitespaces).lowercased()
    }
    if let s = lowerMap[key] {
        store.markSnippetUsed(s.phrase)
        return (s.replacement, 1)
    }

    var out = text
    var count = 0
    for s in snippets.sorted(by: { $0.phrase.count > $1.phrase.count }) {
        let phrase = s.phrase.trimmingCharacters(in: .whitespaces)
        if phrase.isEmpty { continue }
        let pattern = "(?<!\\w)" + NSRegularExpression.escapedPattern(for: phrase) + "(?!\\w)"
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
        let ns = out as NSString
        let matches = re.numberOfMatches(in: out, range: NSRange(location: 0, length: ns.length))
        if matches > 0 {
            out = re.stringByReplacingMatches(in: out, range: NSRange(location: 0, length: ns.length),
                                              withTemplate: NSRegularExpression.escapedTemplate(for: s.replacement))
            count += matches
            store.markSnippetUsed(phrase)
        }
    }
    return (out, count)
}

