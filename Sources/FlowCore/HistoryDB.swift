import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// One dictation record, mirrors the history table written by flow.py.
public struct HistoryRecord {
    public var createdAt: String
    public var text: String
    public var rawText: String?
    public var language: String?
    public var appBundle: String?
    public var appName: String?
    public var url: String?
    public var focusedRole: String?
    public var focusedText: String?
    public var audioS: Double?
    public var transcribeS: Double?
    public var cleanupS: Double?
    public var pasteS: Double?
    public var totalS: Double?
    public var charsIn: Int?
    public var charsOut: Int?
    public var backend: String?
    public var model: String?
    public var aiBackend: String?
    public var aiUsed: Int?
    public var aiKept: Int?
    public var snippetCount: Int
    public var pasteOk: Int?
    public var pasteMethod: String?
    public var pasteError: String?

    public init(createdAt: String, text: String) {
        self.createdAt = createdAt
        self.text = text
        self.snippetCount = 0
    }
}

/// Same DB and schema as the Python app: ~/.flow/flow.sqlite, table history.
public final class HistoryDB {
    public static let dbPath = FlowConfig.flowDir.appendingPathComponent("flow.sqlite").path
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "flow.history")

    public init?() {
        guard sqlite3_open(Self.dbPath, &db) == SQLITE_OK else { return nil }
        let schema = """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            raw_text TEXT,
            language TEXT,
            app_bundle TEXT,
            app_name TEXT,
            url TEXT,
            focused_role TEXT,
            focused_text TEXT,
            audio_s REAL,
            transcribe_s REAL,
            cleanup_s REAL,
            paste_s REAL,
            total_s REAL,
            chars_in INTEGER,
            chars_out INTEGER,
            backend TEXT,
            model TEXT,
            ai_backend TEXT,
            ai_used INTEGER,
            ai_kept INTEGER,
            snippet_count INTEGER DEFAULT 0,
            paste_ok INTEGER,
            paste_method TEXT,
            paste_error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_history_app ON history(app_bundle, created_at DESC);
        """
        sqlite3_exec(db, schema, nil, nil, nil)
    }

    deinit { sqlite3_close(db) }

    public func insert(_ r: HistoryRecord) {
        queue.sync {
            let sql = """
            INSERT INTO history (created_at, text, raw_text, language, app_bundle, app_name,
                url, focused_role, focused_text, audio_s, transcribe_s, cleanup_s, paste_s,
                total_s, chars_in, chars_out, backend, model, ai_backend, ai_used, ai_kept,
                snippet_count, paste_ok, paste_method, paste_error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }

            func bindText(_ i: Int32, _ s: String?) {
                if let s { sqlite3_bind_text(stmt, i, s, -1, SQLITE_TRANSIENT) }
                else { sqlite3_bind_null(stmt, i) }
            }
            func bindReal(_ i: Int32, _ d: Double?) {
                if let d { sqlite3_bind_double(stmt, i, d) } else { sqlite3_bind_null(stmt, i) }
            }
            func bindInt(_ i: Int32, _ n: Int?) {
                if let n { sqlite3_bind_int64(stmt, i, Int64(n)) } else { sqlite3_bind_null(stmt, i) }
            }

            bindText(1, r.createdAt)
            bindText(2, r.text)
            bindText(3, r.rawText)
            bindText(4, r.language)
            bindText(5, r.appBundle)
            bindText(6, r.appName)
            bindText(7, r.url)
            bindText(8, r.focusedRole)
            bindText(9, r.focusedText)
            bindReal(10, r.audioS)
            bindReal(11, r.transcribeS)
            bindReal(12, r.cleanupS)
            bindReal(13, r.pasteS)
            bindReal(14, r.totalS)
            bindInt(15, r.charsIn)
            bindInt(16, r.charsOut)
            bindText(17, r.backend)
            bindText(18, r.model)
            bindText(19, r.aiBackend)
            bindInt(20, r.aiUsed)
            bindInt(21, r.aiKept)
            bindInt(22, r.snippetCount)
            bindInt(23, r.pasteOk)
            bindText(24, r.pasteMethod)
            bindText(25, r.pasteError)
            sqlite3_step(stmt)
        }
    }

    // MARK: - Snippets (same table as flow.py FlowStore)

    public struct Snippet {
        public let phrase: String
        public let replacement: String
        public let useCount: Int
    }

    public func listSnippets() -> [Snippet] {
        queue.sync {
            var out: [Snippet] = []
            var stmt: OpaquePointer?
            let sql = """
            SELECT phrase, replacement, use_count FROM snippets
            WHERE enabled = 1 ORDER BY use_count DESC, phrase ASC
            """
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return out }
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                let p = sqlite3_column_text(stmt, 0).map { String(cString: $0) } ?? ""
                let r = sqlite3_column_text(stmt, 1).map { String(cString: $0) } ?? ""
                let c = Int(sqlite3_column_int64(stmt, 2))
                out.append(Snippet(phrase: p, replacement: r, useCount: c))
            }
            return out
        }
    }

    public func markSnippetUsed(_ phrase: String) {
        queue.sync {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "UPDATE snippets SET use_count = use_count + 1 WHERE phrase = ?", -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, phrase, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
    }

    public func recentTranscripts(limit: Int = 1500) -> [String] {
        queue.sync {
            var result: [String] = []
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT text FROM history WHERE text IS NOT NULL AND text != '' ORDER BY id DESC LIMIT ?", -1, &stmt, nil) == SQLITE_OK else { return result }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, Int32(clamping: max(0, limit)))
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let value = sqlite3_column_text(stmt, 0) { result.append(String(cString: value)) }
            }
            return result
        }
    }

    public func clearHistory() {
        queue.sync { sqlite3_exec(db, "DELETE FROM history", nil, nil, nil) }
    }

    public struct Stats {
        public var count = 0
        public var sumChars = 0
        public var avgTotal = 0.0
        public var topApps: [(String, Int)] = []
        public var snippetCount = 0

    }

    public func stats() -> Stats {
        queue.sync {
            var s = Stats()
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT COUNT(*), COALESCE(SUM(chars_out),0), COALESCE(AVG(total_s),0) FROM history", -1, &stmt, nil) == SQLITE_OK {
                if sqlite3_step(stmt) == SQLITE_ROW {
                    s.count = Int(sqlite3_column_int64(stmt, 0))
                    s.sumChars = Int(sqlite3_column_int64(stmt, 1))
                    s.avgTotal = sqlite3_column_double(stmt, 2)
                }
                sqlite3_finalize(stmt)
            }
            if sqlite3_prepare_v2(db, "SELECT COALESCE(app_name, app_bundle, '?'), COUNT(*) c FROM history GROUP BY 1 ORDER BY c DESC LIMIT 5", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let name = sqlite3_column_text(stmt, 0).map { String(cString: $0) } ?? "?"
                    s.topApps.append((name, Int(sqlite3_column_int64(stmt, 1))))
                }
                sqlite3_finalize(stmt)
            }
            if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM snippets WHERE enabled = 1", -1, &stmt, nil) == SQLITE_OK {
                if sqlite3_step(stmt) == SQLITE_ROW {
                    s.snippetCount = Int(sqlite3_column_int64(stmt, 0))
                }
                sqlite3_finalize(stmt)
            }
            return s
        }
    }

    /// Latest rows for the History menu (created_at, text), newest first.
    public func recent(limit: Int = 10) -> [(String, String)] {
        queue.sync {
            var out: [(String, String)] = []
            var stmt: OpaquePointer?
            let sql = "SELECT created_at, text FROM history ORDER BY id DESC LIMIT ?"
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return out }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int64(stmt, 1, Int64(limit))
            while sqlite3_step(stmt) == SQLITE_ROW {
                let ts = sqlite3_column_text(stmt, 0).map { String(cString: $0) } ?? ""
                let tx = sqlite3_column_text(stmt, 1).map { String(cString: $0) } ?? ""
                out.append((ts, tx))
            }
            return out
        }
    }
}
