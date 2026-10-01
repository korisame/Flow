import Foundation

/// Port of flow.py config handling. Same file, same keys: ~/.flow/config.json
public final class FlowConfig {
    public static let flowDir = ProcessInfo.processInfo.environment["FLOW_DATA_DIR"].map {
        URL(fileURLWithPath: $0, isDirectory: true)
    } ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".flow", isDirectory: true)
    public static let configPath = flowDir.appendingPathComponent("config.json")

    private var data: [String: Any]
    private let queue = DispatchQueue(label: "flow.config")

    public static let defaults: [String: Any] = [
        "backend": "fluidaudio",
        "model": Transcriber.selectedModelLabel,
        "cleanup_model": "flow-fullstop-base",
        "sound_feedback": true,
        "remove_fillers": true,
        "verbal_commands": true,
        "ai_backend": "local",
        "ai_tone": "neutral",
        "show_hud": true,
        "verify_paste": true,
        "use_context": true,
        "snippet_expansion": true,
        "app_tone_overrides": [String: Any](),
        "user_dictionary": [String](),
    ]

    public init() {
        var loaded: [String: Any] = [:]
        var shouldPersistMigration = false
        if let raw = try? Data(contentsOf: Self.configPath),
           let obj = try? JSONSerialization.jsonObject(with: raw) as? [String: Any] {
            loaded = obj
        }
        var merged = Self.defaults
        for (k, v) in loaded { merged[k] = v }
        // migration: legacy cloud backends collapse to local (as in flow.py load_config)
        if let ab = merged["ai_backend"] as? String,
           ["codex", "auto", "openai", "haiku"].contains(ab) {
            merged["ai_backend"] = "local"
        }
        if let backend = merged["backend"] as? String,
           ["parakeet-mlx", "parakeet-coreml"].contains(backend) {
            merged["backend"] = "fluidaudio"
            shouldPersistMigration = true
        }
        if let model = merged["model"] as? String,
           ["parakeet-tdt-0.6b-v3-coreml", "FluidInference/parakeet-tdt-0.6b-v3-coreml"].contains(model) {
            merged["model"] = Transcriber.selectedModelLabel
            shouldPersistMigration = true
        }
        // Full-auto language: the manual language setting is gone, Flow always
        // detects the spoken language on its own. Drop any legacy key.
        merged.removeValue(forKey: "language")
        self.data = merged
        if shouldPersistMigration { save() }
    }

    public func bool(_ key: String, default def: Bool = false) -> Bool {
        queue.sync { (data[key] as? Bool) ?? def }
    }

    public func string(_ key: String) -> String? {
        queue.sync {
            let v = data[key]
            if v is NSNull { return nil }
            return v as? String
        }
    }

    public func int(_ key: String, default def: Int) -> Int {
        queue.sync {
            if let n = data[key] as? Int { return n }
            if let d = data[key] as? Double { return Int(d) }
            if let s = data[key] as? String, let n = Int(s) { return n }
            return def
        }
    }

    public func stringList(_ key: String) -> [String] {
        queue.sync { (data[key] as? [String]) ?? [] }
    }

    public func set(_ key: String, _ value: Any) {
        queue.sync { data[key] = value }
        save()
    }

    public func dictionary(_ key: String) -> [String: Any] {
        queue.sync { (data[key] as? [String: Any]) ?? [:] }
    }

    public var userDictionary: [String] { stringList("user_dictionary") }
    public var aiBackend: String { string("ai_backend") ?? "local" }
    public var soundFeedback: Bool { bool("sound_feedback", default: true) }
    public var removeFillers: Bool { bool("remove_fillers", default: true) }

    public func save() {
        queue.sync {
            guard let raw = try? JSONSerialization.data(
                withJSONObject: data, options: [.prettyPrinted, .sortedKeys]) else { return }
            try? FileManager.default.createDirectory(at: Self.flowDir, withIntermediateDirectories: true)
            try? raw.write(to: Self.configPath, options: .atomic)
        }
    }
}
