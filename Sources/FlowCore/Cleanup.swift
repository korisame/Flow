import Foundation

// Port of flow.py AICleanup: prompt building, guard rails, meta stripping.
// The actual LLM call is abstracted behind CleanupLLM so the backend can be
// swapped (MLX Swift, subprocess, none).

public protocol CleanupLLM {
    /// Run the chat messages through the model, temp 0, non-thinking.
    /// Returns raw model output or nil on failure.
    func generate(messages: [(role: String, content: String)], maxTokens: Int) -> String?
}

public struct ChatMessage: Sendable, Equatable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public final class AICleanup {
    let cfg: FlowConfig
    public var llm: CleanupLLM?

    public init(cfg: FlowConfig, llm: CleanupLLM? = nil) {
        self.cfg = cfg
        self.llm = llm
    }

    public var backend: String {
        let b = cfg.aiBackend
        if b == "none" || b == "local" { return b }
        return "local"
    }

    public var tone: String { cfg.string("ai_tone") ?? "auto" }
    public func isEnabled() -> Bool { backend != "none" }

    // MARK: - Tone

    static let toneNames = ["auto", "neutral", "casual", "formal", "notes", "code"]

    func effectiveTone(appBundle: String?) -> String {
        if tone != "auto" { return tone }
        guard let appBundle, !appBundle.isEmpty else { return "neutral" }
        // app_tone_overrides
        if let o = cfg.dictionary("app_tone_overrides")[appBundle] as? String,
           Self.toneNames.contains(o), o != "auto" {
            return o
        }
        let b = appBundle.lowercased()
        // "beeper" is the bridge Shaun actually writes WhatsApp and Instagram
        // messages through, so it belongs here even though its bundle id names
        // none of the networks it carries.
        if ["slack", "discord", "messages", "whatsapp", "telegram", "instagram",
            "twitter", "beeper", "signal", "texts.com", "wechat", "line.",
            "messenger", "threema"].contains(where: { b.contains($0) }) { return "casual" }
        if ["mail", "outlook", "spark."].contains(where: { b.contains($0) }) { return "formal" }
        if ["xcode", "vscode", "code", "terminal", "iterm", "jetbrains", "sublime", "neovim"].contains(where: { b.contains($0) }) { return "code" }
        if ["notion", "obsidian", "bear", "craft", "ulysses"].contains(where: { b.contains($0) }) { return "notes" }
        return "neutral"
    }

    private func cfgDictionary(_ key: String) -> [String: Any]? {
        // FlowConfig stores raw JSON; expose the dict via a JSON round trip
        // to stay thread-safe with its queue.
        nil
    }

    // MARK: - Prompt

    static func langName(_ code: String?) -> String {
        let map = ["it": "Italian", "en": "English", "ru": "Russian", "ar": "Arabic",
                   "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
                   "zh": "Chinese", "ja": "Japanese", "ko": "Korean"]
        let c = (code ?? "").lowercased()
        return map[c] ?? (code ?? "the same language as the input")
    }

    static let langLock: [String: String] = [
        "ru": "ВНИМАНИЕ: Текст НА РУССКОМ. Выводи ТОЛЬКО на русском языке. НИКОГДА не переводи на английский или другой язык. Используй кириллицу.",
        "ar": "تنبيه: النص باللغة العربية. أخرج النتيجة باللغة العربية فقط. لا تترجم أبدًا إلى الإنجليزية أو أي لغة أخرى. استخدم الأبجدية العربية.",
        "it": "ATTENZIONE: il testo è in italiano. Restituisci SOLO testo italiano. Non tradurre mai in inglese o altre lingue.",
        "fr": "ATTENTION : le texte est en français. Renvoie UNIQUEMENT du français. Ne traduis jamais en anglais ou dans une autre langue.",
        "es": "ATENCIÓN: el texto está en español. Devuelve SOLO texto en español. Nunca traduzcas al inglés u otro idioma.",
        "de": "ACHTUNG: Der Text ist auf Deutsch. Gib NUR deutschen Text zurück. Übersetze niemals ins Englische oder eine andere Sprache.",
        "pt": "ATENÇÃO: o texto está em português. Devolva APENAS texto em português. Nunca traduza para inglês ou outra língua.",
        "zh": "注意：输入是中文。仅以中文输出。绝不翻译成英文或其他语言。",
        "ja": "注意：入力は日本語です。日本語のみで出力してください。英語など他言語に翻訳しないでください。",
        "ko": "주의: 입력은 한국어입니다. 한국어로만 출력하세요. 영어나 다른 언어로 번역하지 마세요.",
    ]

    func systemPrompt(toneId: String, language: String?) -> String {
        let toneLines = [
            "neutral": "Use a neutral, clear tone.",
            "casual": "Use a casual, conversational messaging tone. Short sentences. Contractions OK.",
            "formal": "Use a formal email tone. Complete sentences. No contractions.",
            "notes": "Format as clean prose suitable for notes. Keep it concise.",
            "code": "Treat the input as technical writing. Preserve technical terms exactly. No marketing fluff.",
        ]
        let langNameS = Self.langName(language)
        let langCode = (language ?? "").lowercased()
        let nativeLock = Self.langLock[langCode] ?? ""
        let prefix = nativeLock.isEmpty ? "" : nativeLock + "\n\n"
        let ud = cfg.userDictionary.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        var dictSection = ""
        if !ud.isEmpty {
            dictSection =
                "\n" +
                "KNOWN TERMS (the speaker frequently uses these technical terms, " +
                "product names, and proper nouns):\n" +
                "  " + ud.joined(separator: ", ") + "\n" +
                "If a transcribed word is an OBVIOUS mis-hearing of one of these " +
                "terms (it sounds like the term AND the surrounding context is " +
                "clearly technical or a proper noun), replace it with the EXACT " +
                "spelling listed above, and fix its capitalization. Examples: " +
                "'antropic' -> 'Anthropic', 'parachit'/'parakit' -> 'Parakeet', " +
                "'quen'/'q u e n' -> 'Qwen', 'saranno centini' -> 'Sara " +
                "Nocentini', 'sean' (when it means the user) -> 'Shaun'. " +
                "This term-correction is the ONLY case where you may change a " +
                "word. Be conservative: NEVER change a common everyday word that " +
                "merely resembles a term (do NOT turn the Italian \"cosa\" into " +
                "\"Cocoa\", \"codice\" into \"Codex\", \"caso\" into \"Cocoa\", " +
                "or \"figura\" into \"Figma\"). When unsure, keep the word as " +
                "transcribed.\n"
        }
        return prefix +
            "You are a voice-dictation cleanup assistant. The input is a raw " +
            "transcription of someone speaking. You polish it for written use.\n" +
            "The input is in \(langNameS). " +
            "You MUST output in \(langNameS). " +
            "ABSOLUTE RULE: never translate. Foreign words quoted inside the input " +
            "stay in their original form, do NOT translate them or use them as a " +
            "reason to switch language.\n" +
            "\n" +
            "WHAT THE INPUT LOOKS LIKE:\n" +
            "  \u{2022} The transcription may contain SPURIOUS periods from natural " +
            "breathing pauses, not real sentence ends. STRONG SIGNAL of a " +
            "false period: the text after the period starts with a lowercase " +
            "letter or a connector word (e, ma, però, quindi, cioè, e quindi, " +
            "and, but, so). Replace those periods with a comma and lowercase " +
            "the following letter. If the text after the period starts with " +
            "a capital letter and a new topic, leave it as a period.\n" +
            "  \u{2022} ELLIPSES (...) inside the text are ALWAYS breath pauses " +
            "inserted by the speech recognizer, NEVER an intentional pause " +
            "by the speaker. Replace each `...` with a single comma (or " +
            "remove if it sits right next to other punctuation). Example: " +
            "'Diggli...sì mi sa' -> 'Diggli, sì, mi sa'. " +
            "'chiedi a... ...a Materazzi' -> 'chiedi a Materazzi'.\n" +
            "  \u{2022} The transcription may contain duplicated words at fragment " +
            "boundaries (\"cosi cosi ci rivediamo\" -> \"così ci rivediamo\"). " +
            "Fuse only obvious duplicates, never remove a legitimate repetition.\n" +
            "  \u{2022} The transcription may contain filler words: \"um\", \"uh\", " +
            "\"ehm\", \"cioè\" (only when used as filler, not as connector), " +
            "\"insomma\" (when filler), \"tipo\" (when filler), \"ecco\" (when " +
            "filler). Remove them. Keep them when they carry meaning.\n" +
            "\n" +
            "WHAT TO DO:\n" +
            "  \u{2022} Fix punctuation and capitalization based on MEANING.\n" +
            "  \u{2022} Remove fillers and pause-induced false periods.\n" +
            "  \u{2022} Fix obvious word errors.\n" +
            "  \u{2022} Preserve the speaker's exact wording, register, and content.\n" +
            "\n" +
            "STRICTLY DO NOT:\n" +
            "  \u{2022} Preserve EVERY word from the input. Do not delete, replace, or " +
            "merge content words. Your job is punctuation and capitalization, " +
            "NOT content editing. If a word is in the input, it must appear in " +
            "the output (in the same form, possibly recapitalized).\n" +
            "  \u{2022} CRITICAL: leading interjections (Ascolta, Senti, Guarda, " +
            "Allora, Dunque, Ok, Ciao, Salve, Pronto) are CONTENT words the " +
            "speaker said. ALWAYS keep them. Add a comma after them, but " +
            "NEVER drop them.\n" +
            "  \u{2022} Word-count rule: the output must contain at least as many " +
            "content words as the input. If you find yourself shortening, " +
            "stop and just polish punctuation instead.\n" +
            "  \u{2022} Do NOT add new words that aren't already in the input. Do not " +
            "insert transitional connectors that the speaker didn't say.\n" +
            "  \u{2022} Do NOT shorten, summarize, or rephrase. Same words, just polished.\n" +
            "  \u{2022} Do NOT translate. Do NOT change the meaning.\n" +
            "  \u{2022} Do NOT use em-dashes or en-dashes. Use commas, periods, or " +
            "parentheses.\n" +
            "  \u{2022} Do NOT add markdown, quotation marks around the whole thing, " +
            "explanations, or commentary.\n" +
            "\n" +
            "PUNCTUATION:\n" +
            "  \u{2022} Periods and commas by default.\n" +
            "  \u{2022} Exclamation marks ONLY for clearly emphatic content.\n" +
            "  \u{2022} Question marks only for real questions.\n" +
            "  \u{2022} Preserve proper-noun capitalization (Flow, Llama, iPhone, Ferrari).\n" +
            dictSection +
            "\n" +
            "Output ONLY the cleaned text. No preamble, no quotes around it, no " +
            "explanations. Just the cleaned sentence(s).\n" +
            (toneLines[toneId] ?? toneLines["neutral"]!) +
            (nativeLock.isEmpty ? "" : "\n\n" + nativeLock)
    }

    // MARK: - Few-shot examples

    static let fewShot: [String: [(String, String)]] = [
        "it": [
            ("ehm allora oggi vorrei comprare un iphone ma non so se prendere il pro o il pro max",
             "Allora, oggi vorrei comprare un iPhone, ma non so se prendere il Pro o il Pro Max."),
            ("perfetto fatto benissimo l'app funziona benissimo sono al 90 percento",
             "Perfetto, fatto benissimo. L'app funziona benissimo, sono al 90%."),
            ("Allora voglio un'icona dinamica sulla... sulla taskbar che mi fa vedere... ...vedere lo stato di... di avanzamento del mio utilizzo di codex",
             "Allora, voglio un'icona dinamica sulla taskbar che mi fa vedere lo stato di avanzamento del mio utilizzo di Codex."),
            ("ciao cuore fantastico benissimo. sono molto contento. che cosi ci rivediamo",
             "Ciao cuore, fantastico, benissimo. Sono molto contento che così ci rivediamo."),
            ("ascolta hai fatto per caso quei robi che ha chiesto la elena ieri",
             "Ascolta, hai fatto per caso quei robi che ha chiesto la Elena ieri?"),
            ("senti ti volevo dire che la riunione di domani è spostata",
             "Senti, ti volevo dire che la riunione di domani è spostata."),
            ("guarda non lo so cosa fare in questo caso",
             "Guarda, non lo so cosa fare in questo caso."),
        ],
        "en": [
            ("um so I was thinking about uh the project we discussed yesterday",
             "So I was thinking about the project we discussed yesterday."),
            ("hey can you send me that file please",
             "Hey, can you send me that file, please?"),
        ],
        "fr": [
            ("euh alors aujourd'hui je voudrais acheter un nouveau iphone mais je sais pas si prendre le pro ou le pro max",
             "Alors, aujourd'hui je voudrais acheter un nouvel iPhone, mais je ne sais pas si prendre le Pro ou le Pro Max."),
            ("ouais bah voilà l'application marche super bien on est à 90 pourcent",
             "Voilà, l'application marche super bien, on est à 90%."),
        ],
        "ru": [
            ("ну э я думаю что нужно сделать это завтра",
             "Я думаю, что нужно сделать это завтра."),
            ("слушай ты не мог бы прислать мне этот файл пожалуйста",
             "Слушай, ты не мог бы прислать мне этот файл, пожалуйста?"),
            ("эм ну смотри сегодня я хочу купить новый айфон но не знаю про или про макс",
             "Смотри, сегодня я хочу купить новый iPhone, но не знаю, Pro или Pro Max."),
            ("ладно так значит завтра встречаемся в десять часов в офисе хорошо",
             "Ладно, значит завтра встречаемся в десять часов в офисе, хорошо?"),
        ],
        "ar": [
            ("يعني أنا أفكر إنه لازم نسوي هذا بكرة",
             "أنا أفكر أنه يجب أن نفعل هذا غدًا."),
            ("اسمع ممكن ترسل لي هذا الملف من فضلك",
             "اسمع، هل يمكنك أن ترسل لي هذا الملف من فضلك؟"),
        ],
    ]

    static func buildMessages(sysPrompt: String, text: String, lang: String?) -> [(role: String, content: String)] {
        var msgs: [(role: String, content: String)] = [("system", sysPrompt)]
        let examples = fewShot[(lang ?? "").lowercased()] ?? fewShot["en"]!
        for (raw, clean) in examples {
            msgs.append(("user", raw))
            msgs.append(("assistant", clean))
        }
        msgs.append(("user", text))
        return msgs
    }

    // MARK: - Guards (exact ports)

    static func toneDownPunctuation(_ text: String) -> String {
        if text.isEmpty { return text }
        var t = regexReplace(text, pattern: "!+", with: "!")
        guard let first = t.firstIndex(of: "!") else { return t }
        if t.filter({ $0 == "!" }).count <= 1 { return t }
        let after = t.index(after: first)
        let head = String(t[..<after])
        let tail = String(t[after...]).replacingOccurrences(of: "!", with: ".")
        t = head + tail
        return t
    }

    static let startFillers: Set<String> = [
        "ehm", "uhm", "um", "uh", "ah", "eh", "mmm", "mm",
        "tipo", "cioè", "insomma", "diciamo",
    ]

    static let punctStrip = CharacterSet(charactersIn: ",.!?;:'\"()-\u{2014}\u{2013}\u{2026}")

    /// Returns fixed cleaned text, or nil when the AI dropped a whole opening
    /// phrase (caller must fall back to the raw transcript).
    static func preserveLeadingInterjection(original: String, cleaned: String) -> String? {
        if original.isEmpty || cleaned.isEmpty { return cleaned }
        let words = original.trimmingCharacters(in: .whitespaces).split(separator: " ").map(String.init)
        if words.isEmpty { return cleaned }
        var firstWord: String? = nil
        var secondSubstantive: String? = nil
        for w in words {
            let cleanW = w.trimmingCharacters(in: punctStrip).lowercased()
            if cleanW.isEmpty { continue }
            if firstWord == nil {
                if startFillers.contains(cleanW) { continue }
                firstWord = w.trimmingCharacters(in: punctStrip)
                continue
            }
            if cleanW.count >= 4 {
                secondSubstantive = w.trimmingCharacters(in: punctStrip)
                break
            }
        }
        guard let fw = firstWord else { return cleaned }
        let firstLower = fw.lowercased()
        let head = String(cleaned.trimmingCharacters(in: .whitespaces).prefix(60)).lowercased()
        if regexSearch(head, pattern: "\\b" + NSRegularExpression.escapedPattern(for: firstLower) + "\\b") {
            return cleaned
        }
        if let ss = secondSubstantive {
            let secondLower = ss.lowercased()
            if !regexSearch(head, pattern: "\\b" + NSRegularExpression.escapedPattern(for: secondLower) + "\\b") {
                flowLog("[ai] AI dropped opening phrase ('\(fw)' AND '\(ss)' missing), rejecting cleanup, using raw transcript")
                return nil
            }
        }
        var rest = cleaned.trimmingCharacters(in: .whitespaces)
        while rest.hasPrefix(",") { rest.removeFirst() }
        rest = rest.trimmingCharacters(in: .whitespaces)
        if let f = rest.first, f.isLetter, f.isUppercase {
            rest = String(f).lowercased() + rest.dropFirst()
        }
        let fixed = fw.prefix(1).uppercased() + fw.dropFirst().lowercased() + ", " + rest
        flowLog("[ai] re-prepending dropped first word '\(fw.capitalized)'")
        return fixed
    }

    static func looksInvented(original: String, cleaned: String) -> Bool {
        if cleaned.isEmpty || original.isEmpty { return false }
        func toks(_ s: String) -> Set<String> {
            var out = Set<String>()
            guard let re = try? NSRegularExpression(pattern: "[\\w'\u{2019}]+") else { return out }
            let low = s.lowercased() as NSString
            re.enumerateMatches(in: low as String, range: NSRange(location: 0, length: low.length)) { m, _, _ in
                if let m {
                    let t = low.substring(with: m.range)
                    if t.count >= 3 { out.insert(t) }
                }
            }
            return out
        }
        let inT = toks(original), outT = toks(cleaned)
        if inT.isEmpty || outT.isEmpty { return false }
        let overlap = Double(inT.intersection(outT).count) / Double(outT.count)
        return overlap < 0.70
    }

    /// Words the cleanup corrupted: present in the output, absent from the
    /// input, and one letter away from an input word of the same length.
    ///
    /// That is the signature of the model mistyping a word rather than editing
    /// the text: it turned "avrei" into "avhei", which every other guard waved
    /// through because a single letter barely moves word overlap or length.
    ///
    /// Deliberately narrow. Merges ("e mail" -> "email"), splits, deletions and
    /// whole inserted words all have different shapes and are left to the other
    /// guards, because flagging them too rejected 55% of real cleanups.
    public static func alteredWords(original: String, cleaned: String) -> [String] {
        func tokens(_ s: String) -> [String] {
            var out: [String] = []
            guard let re = try? NSRegularExpression(pattern: "[\\p{L}\\p{N}'\u{2019}]+") else { return out }
            let ns = s as NSString
            re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m else { return }
                // Fold case and accents: "perche" -> "perché" is a fix worth
                // keeping, "avrei" -> "avhei" is not.
                let folded = ns.substring(with: m.range)
                    .folding(options: [.diacriticInsensitive, .caseInsensitive],
                             locale: Locale(identifier: "it_IT"))
                    .replacingOccurrences(of: "\u{2019}", with: "'")
                if !folded.isEmpty { out.append(folded) }
            }
            return out
        }
        let inTokens = Set(tokens(original))
        guard !inTokens.isEmpty else { return [] }

        var altered: [String] = []
        for token in Set(tokens(cleaned)) where !inTokens.contains(token) {
            guard token.count >= 4 else { continue }
            let isTypoOfAnInputWord = inTokens.contains { candidate in
                candidate.count == token.count && oneLetterApart(candidate, token)
            }
            if isTypoOfAnInputWord { altered.append(token) }
        }
        return altered.sorted()
    }

    /// Exactly one substituted character, same length. Cheaper and stricter
    /// than a full edit distance, which is the point.
    private static func oneLetterApart(_ a: String, _ b: String) -> Bool {
        let ca = Array(a), cb = Array(b)
        guard ca.count == cb.count else { return false }
        var diffs = 0
        for i in 0..<ca.count where ca[i] != cb[i] {
            diffs += 1
            if diffs > 1 { return false }
        }
        return diffs == 1
    }

    static func droppedContentWords(original: String, cleaned: String) -> Bool {
        if cleaned.isEmpty || original.isEmpty { return false }
        let fillers: Set<String> = ["ehm", "eh", "uh", "um", "mah", "beh", "boh", "cioe", "cioè",
                                    "insomma", "tipo", "ecco", "niente", "diciamo"]
        func nContent(_ s: String) -> Int {
            guard let re = try? NSRegularExpression(pattern: "[0-9a-zàèéìòóùü'\u{2019}]+") else { return 0 }
            let low = s.lowercased() as NSString
            var n = 0
            re.enumerateMatches(in: low as String, range: NSRange(location: 0, length: low.length)) { m, _, _ in
                if let m {
                    let w = low.substring(with: m.range)
                    if !fillers.contains(w) && w.count > 1 { n += 1 }
                }
            }
            return n
        }
        let nr = nContent(original)
        let nc = nContent(cleaned)
        if nr < 12 { return false }
        return nc < nr - max(2, Int(Double(nr) * 0.03))
    }

    static func looksTranslated(original: String, cleaned: String, expectedLang: String?) -> Bool {
        guard !cleaned.isEmpty, let expectedLang, !expectedLang.isEmpty else { return false }
        let e = expectedLang.lowercased()
        if e == "en" { return false }
        let enStops: Set<String> = ["the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could","should",
            "and","or","but","if","so","as","at","in","on","to","of","for",
            "from","with","without","about","this","that","these","those",
            "you","we","they","them","he","she","it","i","my","your","our","their",
            "not","no","yes","well","just","very","really","i'm","let's","it's",
            "don't","doesn't","didn't","won't","can't","wouldn't","shouldn't",
            "see","said","says","say","get","got","go","goes","went","come","came",
            "make","made","know","knew","think","thought","want","wanted",
            "perfectly","listening"]
        let itStops: Set<String> = ["che", "non", "di", "il", "la", "lo", "le", "un", "una", "ho",
            "ha", "hai", "siamo", "sono", "è", "sei", "sì", "se", "per",
            "con", "ma", "anche", "come", "questo", "questa", "quello"]
        let letters = cleaned.unicodeScalars.filter { CharacterSet.letters.contains($0) }
        let nLetters = letters.count
        let ruCyril = letters.filter { (0x0400...0x04FF).contains(Int($0.value)) }.count
        let arLet = letters.filter { (0x0600...0x06FF).contains(Int($0.value)) }.count
        let ruRatio = nLetters > 0 ? Double(ruCyril) / Double(nLetters) : 0
        let arRatio = nLetters > 0 ? Double(arLet) / Double(nLetters) : 0
        let toks = cleaned.split(separator: " ").map {
            $0.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".,!?:;\"'()"))
        }
        if toks.isEmpty { return false }
        let enHits = toks.filter { enStops.contains($0) }.count
        let itHits = toks.filter { itStops.contains($0) }.count
        let enRatio = Double(enHits) / Double(toks.count)
        if e == "ru" && cleaned.count >= 4 {
            if ruCyril == 0 || ruRatio < 0.50 { return true }
        }
        if e == "ar" && cleaned.count >= 4 {
            if arLet == 0 || arRatio < 0.50 { return true }
        }
        if e == "it" && enRatio > 0.18 && enHits > itHits { return true }
        if ["es", "fr", "pt", "de"].contains(e) && enRatio > 0.25 { return true }
        return false
    }

    // MARK: - Meta stripper (port of _strip_meta)

    static func stripMeta(_ out: String, original: String) -> String {
        if out.isEmpty { return out }
        var s = out.trimmingCharacters(in: .whitespacesAndNewlines)

        let leadingHeaders = "^\\s*(?:" +
            "Ecco (?:il |la |un)?[^\\n]{0,80}?[:.]\\s*\\n+|" +
            "Here(?:'s| is)\\s+(?:the\\s+)?[^\\n]{0,80}?[:.]\\s*\\n+|" +
            "Cleaned(?: text)?\\s*:\\s*\\n*|" +
            "Corrected(?: text)?\\s*:\\s*\\n*|" +
            "Output\\s*:\\s*\\n*|" +
            "Risultato\\s*:\\s*\\n*" +
            ")"
        for _ in 0..<3 {
            guard let re = try? NSRegularExpression(pattern: leadingHeaders, options: [.caseInsensitive]) else { break }
            let ns = s as NSString
            guard let m = re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length)) else { break }
            let newS = ns.replacingCharacters(in: m.range, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            if newS == s { break }
            s = newS
        }

        let quotePairs: [(String, String)] = [("\"", "\""), ("'", "'"),
            ("\u{201C}", "\u{201D}"), ("\u{2018}", "\u{2019}"),
            ("\u{00AB}", "\u{00BB}"), ("\u{201E}", "\u{201C}")]
        for (ql, qr) in quotePairs {
            if s.hasPrefix(ql) && s.hasSuffix(qr) && s.count > 2 {
                s = String(s.dropFirst(ql.count).dropLast(qr.count)).trimmingCharacters(in: .whitespaces)
                break
            }
        }

        let trailingPattern = "(?:\\n|\\.\\s+|^)\\s*" +
            "(Ho corretto|Ho rimosso|Ho aggiunto|Ho cambiato|Ho modificato|Ho anche|" +
            "I have corrected|I corrected|I removed|I added|I changed|I rewrote|" +
            "I also |I've |" +
            "Here(?:'s| is)\\s+(?:the\\s+)?(cleaned|corrected|fixed|updated|revised)|" +
            "Cleaned text:|Corrected text:|Original:|Note:|Translation:|" +
            "Spiegazione:|Modifiche:|Cambiamenti:|Esempio:|Nota:|" +
            "In summary[,:]|To summarize[,:]|" +
            "\\(Note|\\(Nota|\\[Note|\\[Nota)"
        if let re = try? NSRegularExpression(pattern: trailingPattern, options: [.caseInsensitive]) {
            let ns = s as NSString
            if let m = re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length)) {
                s = ns.substring(to: m.range.location).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        for (ql, qr) in quotePairs {
            if s.hasPrefix(ql) && s.hasSuffix(qr) && s.count > 2 {
                s = String(s.dropFirst(ql.count).dropLast(qr.count)).trimmingCharacters(in: .whitespaces)
                break
            }
        }

        if s.count > max(80, Int(Double(original.count) * 1.6)) {
            let firstPara = s.components(separatedBy: "\n\n").first?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !firstPara.isEmpty && firstPara.count <= Int(Double(original.count) * 1.6) {
                s = firstPara
            } else {
                return ""
            }
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Public entry point (port of clean())

    /// Would `clean` actually call the model for this text? Lets the caller
    /// paste the draft first and correct it afterwards only when a real
    /// (multi-second) inference is about to happen.
    public func wouldRunModel(_ input: String, force: Bool = false) -> Bool {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty || !isEnabled() || llm == nil { return false }
        return force || skipReason(for: text) == nil
    }

    /// nil = the model runs; otherwise the reason it was skipped.
    private func skipReason(for text: String) -> String? {
        let wordCount = text.split(separator: " ").count
        if wordCount < 10 { return "short utterance (\(wordCount) words)" }
        let lower = " " + text.lowercased() + " "
        let hasCleanupSignal = [" um ", " uh ", " ehm ", " cioè ", " insomma ",
                                " tipo ", " ecco ", " ... "].contains { lower.contains($0) }
        let looksComplete = (text.first?.isUppercase ?? false)
            && ".!?\u{2026}".contains(text.last ?? " ")
        let shortEnoughToTrust = wordCount <= 28 && text.count <= 220
        if looksComplete && shortEnoughToTrust && !hasCleanupSignal {
            return "already clean-looking (\(wordCount) words, \(text.count) chars)"
        }
        return nil
    }

    public func clean(_ input: String, appBundle: String? = nil,
                      language: String? = nil, force: Bool = false) -> String {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty || !isEnabled() { return text }

        if let punctuation = llm as? LocalCleanupModel {
            if let language, !["it", "en", "fr", "de", "nl"].contains(language) { return text }
            return punctuation.generate(messages: [("user", text)], maxTokens: 0) ?? text
        }

        if !force, let reason = skipReason(for: text) {
            flowLog("[ai] cleanup skipped: \(reason)")
            return text
        }

        guard let llm else { return text }
        let toneId = effectiveTone(appBundle: appBundle)
        let sysPrompt = systemPrompt(toneId: toneId, language: language)
        let msgs = Self.buildMessages(sysPrompt: sysPrompt, text: text, lang: language)
        // Cleanup is an edit, not a generation: the output tracks the input
        // length. Measured on 185 real runs, 87 came back byte-identical and 56
        // changed 1-3 characters, so a 900-token ceiling only bounded runaway
        // output. Keep enough headroom for Italian (~2 tokens/word) and cap the
        // worst case much lower.
        let maxToks = min(480, max(64, text.split(separator: " ").count * 2 + 32))

        let t0 = Date()
        guard var out = llm.generate(messages: msgs, maxTokens: maxToks) else {
            flowLog("[ai] cleanup failed (local): llm returned nil")
            return text
        }
        if let r = out.range(of: "</think>") { out = String(out[r.upperBound...]) }
        out = out.trimmingCharacters(in: .whitespacesAndNewlines)
        flowLog(String(format: "[ai-local] %.2fs (%d->%d chars)", -t0.timeIntervalSinceNow, text.count, out.count))
        var cleaned: String? = Self.stripMeta(out, original: text)

        if let c = cleaned, !c.isEmpty, Self.looksTranslated(original: text, cleaned: c, expectedLang: language) {
            flowLog("[ai] cleanup translated to wrong language (expected \(language ?? "?")), discarding")
            return text
        }
        if let c = cleaned, !c.isEmpty, Self.looksInvented(original: text, cleaned: c) {
            flowLog("[ai] cleanup invented content (word-overlap < 70%), discarding")
            return text
        }
        if let c = cleaned, !c.isEmpty, Self.droppedContentWords(original: text, cleaned: c) {
            flowLog("[ai] cleanup dropped content words (cropping), using raw transcript")
            return text
        }
        if let c = cleaned, !c.isEmpty {
            let altered = Self.alteredWords(original: text, cleaned: c)
            if !altered.isEmpty {
                flowLog("[ai] cleanup rewrote words \(altered.prefix(6)), using raw transcript")
                return text
            }
        }
        if let c = cleaned, !c.isEmpty {
            let toned = Self.toneDownPunctuation(c)
            guard let preserved = Self.preserveLeadingInterjection(original: text, cleaned: toned) else {
                return text
            }
            cleaned = preserved
        }
        guard let final = cleaned, !final.isEmpty else {
            flowLog("[ai] cleanup output rejected (too inflated), using raw transcript")
            return text
        }
        return final
    }
}

public func flowLog(_ msg: String) {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
}
