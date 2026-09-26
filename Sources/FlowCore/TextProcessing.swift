import Foundation

// Port of flow.py text processing: dictionary correction, verbal commands,
// filler removal, typographic normalization, repeat-loop stripping.

// Known ASR misspellings -> canonical spelling. Exact-match only (keys are
// non-words), so safe. Extend as recurring mis-hearings show up.
public let DICT_ALIASES: [String: String] = [
    "parakit": "Parakeet",
    "parakeet": "Parakeet",
    "parachit": "Parakeet",
    "ermes": "Hermes",
    "razziella": "Graziella",
    "shoom": "showroom",
    "polce": "Porsche",
    "etoro": "eToro",
    "suetoro": "su eToro",
    "gbt": "GPT",
    "toielleria": "gioielleria",
]

private func editDistance(_ a: String, _ b: String, maxd: Int) -> Int {
    let ca = Array(a), cb = Array(b)
    let la = ca.count, lb = cb.count
    if abs(la - lb) > maxd { return maxd + 1 }
    var prev = Array(0...lb)
    for i in 1...max(la, 1) {
        if la == 0 { break }
        var cur = [Int](repeating: 0, count: lb + 1)
        cur[0] = i
        var rowBest = cur[0]
        let ai = ca[i - 1]
        for k in 1...max(lb, 1) {
            if lb == 0 { break }
            let cost = ai == cb[k - 1] ? 0 : 1
            cur[k] = min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + cost)
            if cur[k] < rowBest { rowBest = cur[k] }
        }
        if rowBest > maxd { return maxd + 1 }
        prev = cur
    }
    return prev[lb]
}

/// Ordinary words the ASR gets right and that sit close to a dictionary term.
/// Never rewrite these, however good the edit distance looks.
private let FUZZY_BLOCKLIST: Set<String> = [
    // Ordinary Italian words one edit away from a dictionary term. "carrello"
    // (shopping cart) vs the name "Carrelli" is the one that actually bit:
    // 8 rewrites across the real history before it was blocked.
    "carrello", "carrelli", "gioielliere", "gioiellerie",
    "come", "cosa", "casa", "core", "cara", "caro", "capo", "colla", "colpa",
    "flusso", "flow", "gioco", "loro", "lavoro", "libro", "modo", "molto",
    "nota", "note", "parla", "parte", "porta", "posta", "primo", "prima",
    "quello", "quella", "sara", "sarà", "scelta", "senso", "serio", "sopra",
    "stato", "stessa", "swift", "tempo", "testo", "tutto", "vero", "vista",
    "python", "claude", "codex", "flora", "clara", "chiara", "grazie",
]

/// Dictionary terms that are also everyday lowercase words. Canonicalising
/// these would turn "subito" (the adverb, "right away") into the marketplace
/// name, so only fix them when the user already wrote them capitalised.
/// ("carrelli" is in the dictionary as a surname, but in the real history it
/// always meant boat trailers.)
private let AMBIGUOUS_LOWERCASE: Set<String> = ["subito", "air", "photon", "carrelli"]

/// Fuzzy match for a SINGLE word against the dictionary. Deliberately strict:
/// the whole point of the dictionary is proper nouns and jargon the ASR mangles
/// (Anthropic, Braccialini, Qwen), and a false positive silently rewrites a word
/// the user actually said. Requires a shared first letter, a length within one
/// of the target, and an edit distance under a fifth of the term length.
private func fuzzySingleWord(_ low: String, canon: [String: String],
                             canonNWords: [String: Int]) -> String? {
    guard low.count >= 5, !FUZZY_BLOCKLIST.contains(low) else { return nil }
    guard let first = low.first else { return nil }
    var bestKey: String? = nil
    var bestD = 99
    for (key, _) in canon {
        guard canonNWords[key] == 1, key.count >= 5, key.first == first else { continue }
        guard abs(key.count - low.count) <= 1 else { continue }
        // One edit, always. Two was enough to turn "gioielliere" (jeweller)
        // into "gioielleria" (the shop), so the second edit is only earned by
        // very long terms where a coincidence is implausible.
        let maxd = key.count >= 12 ? 2 : 1
        let d = editDistance(low, key, maxd: maxd)
        if d <= maxd && d < bestD { bestKey = key; bestD = d }
    }
    guard let bk = bestKey else { return nil }
    return canon[bk]
}

/// Fix ASR mis-transcriptions of known technical terms / proper nouns using
/// the user dictionary. Same layers as flow.py: spelled-out collapse, exact
/// case-fix, multi-word edit-distance, boundary repair. Single words match
/// EXACT only (no fuzzy) to avoid cosa->Cocoa style false positives.
public func correctDictionaryTerms(_ text: String, terms: [String]) -> String {
    if text.isEmpty { return text }
    var canon: [String: String] = [:]
    var canonNWords: [String: Int] = [:]
    var maxWords = 1
    for t0 in terms {
        let t = t0.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { continue }
        let key = t.lowercased().replacingOccurrences(of: " ", with: "")
        canon[key] = t
        canonNWords[key] = t.split(separator: " ").count
        maxWords = max(maxWords, t.split(separator: " ").count)
    }
    for (alias, canonical) in DICT_ALIASES {
        let k = alias.lowercased().replacingOccurrences(of: " ", with: "")
        if canon[k] == nil {
            canon[k] = canonical
            canonNWords[k] = canonical.split(separator: " ").count
        }
        maxWords = max(maxWords, alias.split(separator: " ").count)
    }
    if canon.isEmpty { return text }
    let keys = Array(canon.keys)

    func lookup(_ s: String, fuzzy: Bool = true, nwords: Int = 1) -> String? {
        let low = s.lowercased()
        if let hit = canon[low] {
            // Don't promote an everyday lowercase word to its proper-noun form.
            if nwords == 1, AMBIGUOUS_LOWERCASE.contains(low),
               !(s.first?.isUppercase ?? false) {
                return nil
            }
            // The dictionary entry is lowercase but the user capitalised it:
            // it starts a sentence. Leave it alone rather than downcasing.
            if hit.lowercased() == low, (s.first?.isUppercase ?? false),
               !(hit.first?.isUppercase ?? false) {
                return nil
            }
            return hit
        }
        if !fuzzy { return nil }
        if nwords >= 2 && low.count >= 6 {
            var bestKey: String? = nil
            var bestD = 99
            let lowFirst = low.first
            for key in keys {
                if canonNWords[key] != nwords || key.first != lowFirst { continue }
                let d = editDistance(low, key, maxd: 2)
                if d <= 2 && d < bestD { bestKey = key; bestD = d }
            }
            if let bk = bestKey { return canon[bk] }
        }
        if nwords == 1, let hit = fuzzySingleWord(low, canon: canon, canonNWords: canonNWords) {
            return hit
        }
        return nil
    }

    // Collapse spelled-out sequences "q u e n" -> "quen"
    var work = regexReplace(text, pattern: "\\b(?:[A-Za-z] ){2,}[A-Za-z]\\b") { m in
        m.replacingOccurrences(of: " ", with: "")
    }

    // Tokenize into alternating word / non-word parts (unicode letters)
    var parts: [String] = []
    do {
        let re = try NSRegularExpression(pattern: "[^\\W\\d_]+|[\\W\\d_]+",
                                         options: [.useUnicodeWordBoundaries])
        let ns = work as NSString
        re.enumerateMatches(in: work, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            if let m { parts.append(ns.substring(with: m.range)) }
        }
    } catch { return work }

    let widx = parts.indices.filter { !parts[$0].isEmpty && parts[$0].first!.isLetter }

    var spans: [(Int, Int, String)] = []
    var j = 0
    while j < widx.count {
        let i0 = widx[j]
        var run = [i0]
        var k = j + 1
        while parts[i0].count == 1 && k < widx.count && parts[widx[k]].count == 1 {
            run.append(widx[k]); k += 1
        }
        if parts[i0].count == 1 && run.count >= 3 {
            var done = false
            var take = run.count
            while take > 2 {
                let joined = run[0..<take].map { parts[$0] }.joined()
                if let hit = lookup(joined) {
                    spans.append((run[0], run[take - 1], hit))
                    j = j + take
                    done = true
                    break
                }
                take -= 1
            }
            if done { continue }
            j += 1
            continue
        }
        var matched = false
        var n = min(maxWords, widx.count - j)
        while n > 0 {
            let window = Array(widx[j..<(j + n)])
            let joined = window.map { parts[$0] }.joined()
            if let hit = lookup(joined, nwords: n) {
                spans.append((window[0], window[window.count - 1], hit))
                j += n
                matched = true
                break
            }
            n -= 1
        }
        if !matched { j += 1 }
    }

    if spans.isEmpty { return work }
    var spanStart: [Int: (Int, Int, String)] = [:]
    for s in spans { spanStart[s.0] = s }
    var out: [String] = []
    var i = 0
    while i < parts.count {
        if let (_, end, repl) = spanStart[i] {
            out.append(repl)
            i = end + 1
        } else {
            out.append(parts[i])
            i += 1
        }
    }
    work = out.joined()
    return work
}

// MARK: - Fillers and verbal commands

public let FILLER_WORDS = [
    " um ", " uh ", " hmm ", " umm ", " err ",
    " um,", " uh,", " hmm,", " like like ",
]

public let DELETE_SENTINEL = "\u{0}DEL"

/// Verbal commands processed via regex substitution (checked in this order).
public let VERBAL_CMDS: [(String, String)] = [
    ("\\b(delete that|scratch that|cancel that|cancella|annulla|supprime ça|annule ça|efface ça|удали|отмени|احذف ذلك|امسح)\\b", DELETE_SENTINEL),
    ("\\b(new paragraph|nuovo paragrafo|nouveau paragraphe|новый абзац|فقرة جديدة)\\b", "\n\n"),
    ("\\b(new line|a capo|nuova riga|nuova linea|à la ligne|nouvelle ligne|новая строка|سطر جديد)\\b", "\n"),
    ("\\b(exclamation mark|punto esclamativo|point d'exclamation|восклицательный знак|علامة تعجب)\\b", "!"),
    ("\\b(question mark|punto interrogativo|point d'interrogation|вопросительный знак|علامة استفهام)\\b", "?"),
    ("\\b(ellipsis|punti di sospensione|points de suspension|многоточие|نقاط الحذف)\\b", "\u{2026}"),
    ("\\b(em dash|trattino lungo|tiret cadratin|длинное тире|شرطة)\\b", "\u{2014}"),
    ("\\b(open parenthesis|apri parentesi|ouvrir parenthèse|открыть скобку|قوس مفتوح)\\b", "("),
    ("\\b(close parenthesis|chiudi parentesi|fermer parenthèse|закрыть скобку|قوس مغلق)\\b", ")"),
    ("\\b(open quote|virgolette aperte|guillemet ouvert|открыть кавычки|فتح اقتباس)\\b", "\u{201C}"),
    ("\\b(close quote|virgolette chiuse|guillemet fermé|закрыть кавычки|إغلاق اقتباس)\\b", "\u{201D}"),
    ("\\b(semicolon|punto e virgola|point-virgule|точка с запятой|فاصلة منقوطة)\\b", ";"),
    ("\\b(colon|due punti|deux points|двоеточие|نقطتان)\\b", ":"),
    ("\\b(periodo|punto fermo)\\b", "."),
    ("\\b(точка)\\b", "."),
    ("\\b(نقطة)\\b", "."),
    ("\\b(virgola)\\b", ","),
    ("\\b(запятая)\\b", ","),
    ("\\b(فاصلة)\\b", ","),
]

// MARK: - Regex helpers

func regexReplace(_ s: String, pattern: String, options: NSRegularExpression.Options = [],
                  with template: String) -> String {
    guard let re = try? NSRegularExpression(pattern: pattern, options: options) else { return s }
    let ns = s as NSString
    return re.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: ns.length),
                                       withTemplate: template)
}

func regexReplace(_ s: String, pattern: String, options: NSRegularExpression.Options = [],
                  transform: (String) -> String) -> String {
    guard let re = try? NSRegularExpression(pattern: pattern, options: options) else { return s }
    let ns = s as NSString
    var result = ""
    var last = 0
    re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
        guard let m else { return }
        result += ns.substring(with: NSRange(location: last, length: m.range.location - last))
        result += transform(ns.substring(with: m.range))
        last = m.range.location + m.range.length
    }
    result += ns.substring(from: last)
    return result
}

func regexSearch(_ s: String, pattern: String, options: NSRegularExpression.Options = []) -> Bool {
    guard let re = try? NSRegularExpression(pattern: pattern, options: options) else { return false }
    let ns = s as NSString
    return re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length)) != nil
}

// MARK: - Main text pipeline (port of _process_text)

public func processText(_ input: String, verbalCommands: Bool, removeFillers: Bool) -> String {
    if input.isEmpty { return "" }
    var text = input

    // 0. Normalize typographic punctuation
    let typoMap: [(String, String)] = [
        ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{201E}", "\""),
        ("\u{00AB}", "\""), ("\u{00BB}", "\""),
        ("\u{2018}", "'"), ("\u{2019}", "'"),
        ("\u{2026}", "..."), ("\u{00A0}", " "),
    ]
    for (from, to) in typoMap { text = text.replacingOccurrences(of: from, with: to) }

    // 0b. Strip stand-alone dashes (pause artifacts), keep intra-word hyphens
    text = regexReplace(text, pattern: "\\s+[\u{2014}\u{2013}\u{2212}\\-]+\\s+", with: " ")
    text = regexReplace(text, pattern: "^[\u{2014}\u{2013}\u{2212}\\-]+\\s+", with: "")
    text = regexReplace(text, pattern: "\\s+[\u{2014}\u{2013}\u{2212}\\-]+$", with: "")

    // 1. Verbal commands
    if verbalCommands {
        for (pattern, replacement) in VERBAL_CMDS {
            text = regexReplace(text, pattern: pattern, options: [.caseInsensitive],
                                with: NSRegularExpression.escapedTemplate(for: replacement))
        }
        if text.trimmingCharacters(in: .whitespacesAndNewlines) == DELETE_SENTINEL {
            return DELETE_SENTINEL
        }
        text = text.replacingOccurrences(of: DELETE_SENTINEL, with: "")
    }

    // 2. Filler words
    if removeFillers {
        for w in FILLER_WORDS { text = text.replacingOccurrences(of: w, with: " ") }
    }

    // 3. Normalize whitespace (preserve intentional newlines)
    text = text.split(separator: "\n", omittingEmptySubsequences: false)
        .map { $0.split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ") }
        .joined(separator: "\n")

    text = regexReplace(text, pattern: "\\s+([,.;:!?])", with: "$1")
    text = regexReplace(text, pattern: "([(\\[])\\s+", with: "$1")
    text = regexReplace(text, pattern: "\\s+([)\\]])", with: "$1")

    // 4. Capitalize after sentence-ending punctuation
    text = regexReplace(text, pattern: "(?<=[.!?]\\s)([a-záàâãéèêíïóôõöúçñü])") { m in
        m.uppercased()
    }

    return text.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Repeat-loop stripper (port of _strip_repeat_loop)

public func stripRepeatLoop(_ text: String) -> String {
    if text.isEmpty || text.count < 20 { return text }
    func norm(_ tok: Substring) -> String {
        tok.trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:'\"()[]\u{2026}\u{2014}\u{2013}-")).lowercased()
    }
    var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
    for _ in 0..<8 {
        let words = s.split(separator: " ").map(String.init)
        if words.count < 6 { break }
        let normed = words.map { norm(Substring($0)) }
        var best: (Int, Int, Int)? = nil  // (start, ngramSize, runCount)
        for n in 1...6 {
            let minRun = n == 1 ? 4 : 3
            var i = 0
            while i + minRun * n <= normed.count {
                if (i..<(i + n)).contains(where: { normed[$0].isEmpty }) { i += 1; continue }
                var run = 1
                var j = i + n
                while j + n <= normed.count && Array(normed[j..<(j + n)]) == Array(normed[i..<(i + n)]) {
                    run += 1
                    j += n
                }
                if run >= minRun {
                    if best == nil || run * n > best!.2 * best!.1 {
                        best = (i, n, run)
                    }
                    i = j
                } else {
                    i += 1
                }
            }
        }
        guard let (start, n, run) = best else { break }
        let end = start + n * run
        var newWords: [String]
        if end >= words.count - 2 {
            newWords = Array(words[0..<start])
        } else {
            newWords = Array(words[0..<(start + n)]) + Array(words[end...])
        }
        var newS = newWords.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        if let lastCh = newS.last, !".!?\u{2026}".contains(lastCh) {
            newS += "."
        }
        if newS == s { break }
        FileHandle.standardError.write("[transcribe] repeat-loop stripped: n=\(n) run=\(run) start=\(start)\n".data(using: .utf8)!)
        s = newS
    }
    return s
}

