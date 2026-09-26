import AppKit
import Observation
import FlowCore

/// Ponte osservabile tra l'AppDelegate (che resta il proprietario della pipeline)
/// e le viste SwiftUI di barra, HUD e pannello.
/// Va toccato solo dal thread principale: l'AppDelegate lo aggiorna sempre da li'.
@Observable
final class FlowUIModel {
    weak var app: AppDelegate?

    var state: FlowState = .loading
    var statusLine: String = "Caricamento modello…"
    var recordingSince: Date?
    var micLevel: Double = 0
    var preview: String = ""
    var history: [(stamp: String, text: String)] = []
    var stats: (count: Int, words: Int, avg: Double) = (0, 0, 0)
    var lastError: String?

    private var levelTimer: Timer?

    init(app: AppDelegate? = nil) {
        self.app = app
    }

    // MARK: - Aggiornamenti dallo strato AppKit

    func apply(legacyState: String) {
        let new = FlowState(legacy: legacyState)
        state = new
        if new.isLive {
            if recordingSince == nil { recordingSince = .now }
            startLevelPolling()
        } else {
            recordingSince = nil
            stopLevelPolling()
            micLevel = 0
        }
    }

    func reloadHistory() {
        guard let store = app?.store else { return }
        history = store.recent(limit: 10).map { (stamp: $0.0, text: $0.1) }
        let s = store.stats()
        stats = (s.count, s.sumChars / 5, s.avgTotal)
    }

    private func startLevelPolling() {
        guard levelTimer == nil else { return }
        levelTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { [weak self] _ in
            guard let self, let rec = self.app?.recorder else { return }
            self.micLevel = Double(rec.inputLevel)
        }
    }

    private func stopLevelPolling() {
        levelTimer?.invalidate()
        levelTimer = nil
    }

    // MARK: - Config
    // Copie osservabili: FlowConfig non e' osservabile, quindi le viste leggono
    // queste e ogni scrittura passa da qui.

    var aiBackend: String = "local"
    var aiTone: String = "auto"
    var flags: [String: Bool] = [:]

    func reloadConfig() {
        guard let app else { return }
        aiBackend = app.cfg.aiBackend
        aiTone = app.cfg.string("ai_tone") ?? "auto"
        var next: [String: Bool] = [:]
        for row in FlowPanelView.switches {
            next[row.key] = row.key == "launch_at_login"
                ? app.launchAgentInstalled()
                : app.cfg.bool(row.key, default: true)
        }
        flags = next
    }

    func flag(_ key: String, default def: Bool = true) -> Bool { flags[key] ?? def }

    func setBackend(_ id: String) {
        app?.cfg.set("ai_backend", id)
        aiBackend = id
        app?.syncMenuState()
    }

    func setTone(_ id: String) {
        app?.cfg.set("ai_tone", id)
        aiTone = id
        app?.syncMenuState()
    }

    func toggle(_ key: String) {
        let newValue = !flag(key)
        app?.setFlag(key, newValue)
        flags[key] = key == "launch_at_login" ? (app?.launchAgentInstalled() ?? newValue) : newValue
    }

    // MARK: - Azioni

    func pasteLast() {
        guard let text = history.first?.text else { return }
        _ = app?.paster.paste(text)
    }

    func paste(_ text: String) { _ = app?.paster.paste(text) }

    func clearHistory() {
        app?.store?.clearHistory()
        app?.rebuildHistoryMenu()
        reloadHistory()
    }

    func freeMemory() { app?.cbFreeMemory(NSMenuItem()) }
    func reset() { app?.cbEmergencyReset(NSMenuItem()) }
    func openHub() { app?.cbOpenHub(NSMenuItem()) }
    func editDictionary() { app?.cbEditDictionary(NSMenuItem()) }
    func suggestDictionary() { app?.cbSuggestDictionary(NSMenuItem()) }
    func setCurrentAppTone() { app?.cbSetCurrentAppTone(NSMenuItem()) }
    func kofi() { NSWorkspace.shared.open(URL(string: KOFI_URL)!) }
    func restart() { app?.cbRestart(NSMenuItem()) }
    func quit() { NSApp.terminate(nil) }

    func openPermissions() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")!)
    }

    var needsPermission: Bool {
        statusLine.contains("⚠") || statusLine.lowercased().contains("grant")
    }
}

