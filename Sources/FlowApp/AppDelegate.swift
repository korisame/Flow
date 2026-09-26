import AppKit
import SwiftUI
import Foundation
import CoreGraphics
import FlowCore

let APP_NAME = "Flow"
let APP_VERSION = "2.3.0-recovered"
let KOFI_URL = "https://ko-fi.com/shaungori"
let HOTKEY_FLAG: UInt64 = 0x800000       // kCGEventFlagMaskSecondaryFn
let DOUBLE_PRESS_SEC = 0.45
let MAX_HISTORY = 10

let LANGUAGES: [(String, String?)] = [
    ("\u{1F310}  Auto / Multilingual", nil),
    ("\u{1F1EC}\u{1F1E7}  English", "en"),
    ("\u{1F1EE}\u{1F1F9}  Italiano", "it"),
    ("\u{1F1EB}\u{1F1F7}  Français", "fr"),
    ("\u{1F1F7}\u{1F1FA}  Русский", "ru"),
    ("\u{1F1F8}\u{1F1E6}  عربي", "ar"),
]

let AI_BACKENDS: [(String, String)] = [
    ("none", "None  ·  raw transcription"),
    // The label is replaced at menu build time with the model actually loaded
    // (see cleanup_model in config.json).
    ("local", "Local  ·  FullStop, offline"),
]

let AI_TONES: [(String, String)] = [
    ("auto", "Auto  ·  per app"),
    ("neutral", "Neutral"),
    ("casual", "Casual messaging"),
    ("formal", "Formal email"),
    ("notes", "Clean notes"),
    ("code", "Technical / code"),
]

final class AppDelegate: NSObject, NSApplicationDelegate {
    let cfg = FlowConfig()
    let store = HistoryDB()
    let recorder = Recorder()
    let transcriber = Transcriber()
    let paster = Paster()
    lazy var cleanupModel = LocalCleanupModel()
    lazy var ai = AICleanup(cfg: cfg, llm: cleanupModel)

    var statusItem: NSStatusItem!
    var statusMenuItem: NSMenuItem!
    var lastPasteItem: NSMenuItem!
    var aiBackendMenu: NSMenu!
    var aiToneMenu: NSMenu!
    var historyMenu: NSMenu!
    var toggleItems: [String: NSMenuItem] = [:]

    /// Strato SwiftUI: voce nella barra, pannello e HUD leggono da qui.
    /// Il menu AppKit resta costruito e allineato: e' il fallback sul tasto destro.
    let ui = FlowUIModel()
    var statusHost: FlowPassthroughHost<FlowStatusLabel>?
    let panelPopover = NSPopover()
    private var panelDismissMonitor: Any?
    private var legacyMenu: NSMenu?

    var state = "loading"
    var recStart: Date? = nil
    var recTimer: Timer? = nil

    // Hotkey state
    var fnDown = false
    var lastFn: Double = 0
    var deferredStopTimer: DispatchWorkItem? = nil
    var eventTap: CFMachPort? = nil

    // Hotkey edges (press/release/deferred stop/IPC start-stop) all run here so
    // they can never execute out of order or overlap. DispatchQueue.global() is
    // concurrent: a release could overtake — or run alongside — its own press
    // and skip the stop, stranding a recording that the next press then wiped.
    // Never give this queue ASR/LLM/paste work: recorder.stop() already blocks
    // it 0.25s and openEngine() up to 0.8s on a retry.
    let hotkeyQueue = DispatchQueue(label: "flow.hotkey")

    // Worker
    let workQueue = DispatchQueue(label: "flow.worker")
    var pendingItems = 0
    var firstFragmentPasted = false

    /// A run of dictations aimed at the same field with only short pauses
    /// between them: 84 of the last 400 dictations started within 15s of the
    /// previous one. Treating them as one growing text lets the cleanup model
    /// punctuate across the seams instead of seeing disconnected fragments.
    struct BurstState {
        var pid: pid_t
        /// Exactly what Flow has put in the field for this burst so far.
        var textInField: String
        var lastActivity: Date
    }
    var burst: BurstState? = nil
    /// Pause after which the next dictation is a new thought, not a continuation.
    let burstWindowSeconds: Double =
        Double(ProcessInfo.processInfo.environment["FLOW_BURST_WINDOW"] ?? "") ?? 25

    var modelReady = false
    var hud: HudPanel? = nil

    struct IPCRecordingContext {
        let correlationID: String
        let pasteOutput: Bool
    }

    var ipcServer: FlowIPCServer? = nil
    var ipcRecordingContext: IPCRecordingContext? = nil
    private let ipcSessionLock = NSLock()
    private var ipcCancelledCorrelations: Set<String> = []
    private var ipcLatestCorrelationID: String? = nil

    func applicationDidFinishLaunching(_ notification: Notification) {
        let smokeTest = ProcessInfo.processInfo.environment["FLOW_SMOKE_TEST"] == "1"
        setupStatusItem()
        setState("loading")
        updateStatusLine("Loading model…")

        // Accessibility check + prompt
        let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        if !smokeTest && !AXIsProcessTrustedWithOptions(opts) {
            updateStatusLine("⚠️  Grant Accessibility permission — click to open Settings")
        }

        // Load ASR in background
        Task.detached { [self] in
            do {
                try await transcriber.load()
                await MainActor.run {
                    self.modelReady = true
                    self.setState("idle")
                    self.updateStatusLine("Parakeet v3 CoreML · FluidAudio · ready")
                    // Free the ~2.7 GB CoreML/ANE footprint if Flow sits idle
                    // after launch without any dictation.
                    self.scheduleAsrIdleUnload()
                }
                flowLog("[model] ASR ready")
            } catch {
                await MainActor.run {
                    self.updateStatusLine("⚠️  Model failed to load")
                }
                self.notify("Model failed to load", "\(error.localizedDescription)")
            }
        }

        if !smokeTest {
            startEventTapThread()
            setupDebugIPC()
        }
        setupStableIPC()
        let panel = HudPanel()
        panel.recorder = recorder
        hud = panel
    }

    // MARK: - Status item / states

    func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        ui.app = self
        legacyMenu = buildMenu()

        guard let button = statusItem.button else { return }
        let host = FlowPassthroughHost(rootView: FlowStatusLabel(model: ui))
        host.translatesAutoresizingMaskIntoConstraints = false
        button.addSubview(host)
        NSLayoutConstraint.activate([
            host.leadingAnchor.constraint(equalTo: button.leadingAnchor),
            host.trailingAnchor.constraint(equalTo: button.trailingAnchor),
            host.topAnchor.constraint(equalTo: button.topAnchor),
            host.bottomAnchor.constraint(equalTo: button.bottomAnchor),
        ])
        statusHost = host
        button.target = self
        button.action = #selector(statusItemClicked)
        button.sendAction(on: [.leftMouseUp, .rightMouseUp])

        panelPopover.behavior = .transient
        panelPopover.animates = true
        let controller = NSHostingController(rootView: FlowPanelView(model: ui))
        controller.sizingOptions = [.preferredContentSize]
        panelPopover.contentViewController = controller

        syncStatusWidth()
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.syncStatusWidth()
        }
        ui.reloadConfig()
        ui.reloadHistory()
    }

    private func syncStatusWidth() {
        guard let host = statusHost else { return }
        let width = max(24, ceil(host.fittingSize.width))
        if abs(statusItem.length - width) > 0.5 { statusItem.length = width }
    }

    @objc func statusItemClicked() {
        guard let button = statusItem.button else { return }
        // Tasto destro: menu AppKit completo, cosi' nessuna voce storica si perde.
        if NSApp.currentEvent?.type == .rightMouseUp {
            statusItem.menu = legacyMenu
            button.performClick(nil)
            statusItem.menu = nil
            return
        }
        if panelPopover.isShown {
            closePanel()
        } else {
            ui.reloadHistory()
            panelPopover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApp.activate(ignoringOtherApps: true)
            panelPopover.contentViewController?.view.window?.makeKey()
            startPanelDismissWatch()
        }
    }

    func closePanel() {
        stopPanelDismissWatch()
        panelPopover.performClose(nil)
    }

    /// Il solo .transient perde il click fuori dopo un tracking loop (menu, alert):
    /// qui il pannello si chiude sempre, tranne quando il click e' suo.
    private func startPanelDismissWatch() {
        stopPanelDismissWatch()
        panelDismissMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        ) { [weak self] _ in
            guard let self else { return }
            if let frame = self.panelPopover.contentViewController?.view.window?.frame,
               frame.contains(NSEvent.mouseLocation) { return }
            self.closePanel()
        }
        NotificationCenter.default.addObserver(self, selector: #selector(panelAppResigned),
                                               name: NSApplication.didResignActiveNotification, object: nil)
    }

    private func stopPanelDismissWatch() {
        if let panelDismissMonitor { NSEvent.removeMonitor(panelDismissMonitor) }
        panelDismissMonitor = nil
        NotificationCenter.default.removeObserver(self, name: NSApplication.didResignActiveNotification, object: nil)
    }

    @objc private func panelAppResigned() {
        if panelPopover.isShown { closePanel() }
    }

    func applyIcon() {
        // La voce nella barra e' una vista SwiftUI: si aggiorna da sola con lo stato.
        ui.apply(legacyState: state)
    }

    func setState(_ s: String) {
        DispatchQueue.main.async {
            self.state = s
            self.applyIcon()
            if s == "rec" || s == "rec_hf" {
                self.recStart = self.recStart ?? Date()
                self.hud?.show(state: s)
            } else {
                self.recTimer?.invalidate()
                self.recTimer = nil
                self.recStart = nil
                if ["proc", "load", "ai"].contains(s) { self.hud?.show(state: s) }
                else { self.hud?.hide() }
            }
        }
    }

    func updateStatusLine(_ s: String) {
        DispatchQueue.main.async {
            self.statusMenuItem?.title = s
            self.ui.statusLine = s
        }
    }

    /// Applica un interruttore delle impostazioni e tiene allineato il menu AppKit.
    func setFlag(_ key: String, _ value: Bool) {
        if key == "launch_at_login" {
            setLaunchAtLogin(value)
        } else {
            cfg.set(key, value)
            if key == "verify_paste" { paster.verifyPaste = value }
            if key == "free_model_idle" {
                if value { scheduleAsrIdleUnload() }
                else { asrIdleUnloadWork?.cancel(); asrIdleUnloadWork = nil }
            }
        }
        toggleItems[key]?.state = value ? .on : .off
    }

    /// Riporta nel menu AppKit le scelte fatte dal pannello SwiftUI.
    func syncMenuState() {
        let backend = cfg.aiBackend
        for it in aiBackendMenu?.items ?? [] {
            it.state = ((it.representedObject as? String) == backend) ? .on : .off
        }
        let tone = cfg.string("ai_tone") ?? "auto"
        for it in aiToneMenu?.items ?? [] {
            it.state = ((it.representedObject as? String) == tone) ? .on : .off
        }
    }

    // MARK: - Menu

    func buildMenu() -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false

        menu.addItem(disabled("\(APP_NAME) \(APP_VERSION)"))
        menu.addItem(.separator())
        menu.addItem(disabled("Hold Fn — push to talk"))
        menu.addItem(disabled("Press Fn twice — hands-free"))
        menu.addItem(disabled("Language — auto-detected"))
        menu.addItem(.separator())

        lastPasteItem = NSMenuItem(title: "Last paste — none yet",
                                   action: #selector(cbPasteLast(_:)), keyEquivalent: "")
        lastPasteItem.target = self
        menu.addItem(lastPasteItem)

        let histItem = NSMenuItem(title: "History (last 10)", action: nil, keyEquivalent: "")
        historyMenu = NSMenu()
        rebuildHistoryMenu()
        histItem.submenu = historyMenu
        menu.addItem(histItem)

        menu.addItem(.separator())

        let hubItem = NSMenuItem(title: "Open Flow Hub…", action: #selector(cbOpenHub(_:)), keyEquivalent: "")
        hubItem.target = self
        menu.addItem(hubItem)

        let settings = NSMenuItem(title: "Settings", action: nil, keyEquivalent: "")
        let sm = NSMenu()

        let backendItem = NSMenuItem(title: "AI Cleanup", action: nil, keyEquivalent: "")
        aiBackendMenu = NSMenu()
        for (id, label) in AI_BACKENDS {
            // The local backend names the weights actually selected, so the menu
            // never lies about which model is doing the cleanup.
            let shown = (id == "local") ? "Local  ·  \(cleanupModel.modelLabel) (FullStop, offline)" : label
            let it = NSMenuItem(title: shown, action: #selector(cbSetAiBackend(_:)), keyEquivalent: "")
            it.target = self
            it.representedObject = id
            it.state = (id == cfg.aiBackend) ? .on : .off
            aiBackendMenu.addItem(it)
        }
        backendItem.submenu = aiBackendMenu
        sm.addItem(backendItem)

        let toneItem = NSMenuItem(title: "Cleanup Tone", action: nil, keyEquivalent: "")
        aiToneMenu = NSMenu()
        let curTone = cfg.string("ai_tone") ?? "auto"
        for (id, label) in AI_TONES {
            let it = NSMenuItem(title: label, action: #selector(cbSetAiTone(_:)), keyEquivalent: "")
            it.target = self
            it.representedObject = id
            it.state = (id == curTone) ? .on : .off
            aiToneMenu.addItem(it)
        }
        toneItem.submenu = aiToneMenu
        // Generative tone controls do not apply to a punctuation classifier.

        let dictItem = NSMenuItem(title: "Edit Dictionary…", action: #selector(cbEditDictionary(_:)), keyEquivalent: "")
        dictItem.target = self
        sm.addItem(dictItem)
        let suggestItem = NSMenuItem(title: "Suggest Dictionary Terms…",
                                     action: #selector(cbSuggestDictionary(_:)), keyEquivalent: "")
        suggestItem.target = self
        sm.addItem(suggestItem)
        let appTone = NSMenuItem(title: "Set Current App Tone…", action: #selector(cbSetCurrentAppTone(_:)), keyEquivalent: "")
        appTone.target = self
        sm.addItem(appTone)
        sm.addItem(.separator())

        for (key, label) in [("sound_feedback", "Sound Feedback"),
                             ("remove_fillers", "Remove Filler Words"),
                             ("verbal_commands", "Verbal Commands"),
                             ("show_hud", "Show HUD overlay"),
                             ("verify_paste", "Verify paste result"),
                             ("use_context", "Use app context for cleanup"),
                             ("launch_at_login", "Launch at Login")] {
            let it = NSMenuItem(title: label, action: #selector(cbToggle(_:)), keyEquivalent: "")
            it.target = self
            it.representedObject = key
            if key == "launch_at_login" {
                it.state = launchAgentInstalled() ? .on : .off
            } else {
                it.state = cfg.bool(key, default: true) ? .on : .off
            }
            sm.addItem(it)
            toggleItems[key] = it
        }
        // Paste the raw transcript immediately and polish it in place once the
        // cleanup model is done, instead of making the user wait for it.
        let instantItem = NSMenuItem(title: "Instant paste (polish after)",
                                     action: #selector(cbToggle(_:)), keyEquivalent: "")
        instantItem.target = self
        instantItem.representedObject = "instant_paste"
        instantItem.state = cfg.bool("instant_paste", default: true) ? .on : .off
        sm.addItem(instantItem)
        toggleItems["instant_paste"] = instantItem
        // Dictations aimed at the same field within 25s are cleaned up as one
        // text, so punctuation carries across the pause.
        let burstItem = NSMenuItem(title: "Merge quick follow-up dictations",
                                   action: #selector(cbToggle(_:)), keyEquivalent: "")
        burstItem.target = self
        burstItem.representedObject = "merge_bursts"
        burstItem.state = cfg.bool("merge_bursts", default: true) ? .on : .off
        sm.addItem(burstItem)
        toggleItems["merge_bursts"] = burstItem
        // Rolling transcript of the last seconds in the HUD while you talk.
        let previewItem = NSMenuItem(title: "Live preview while dictating",
                                     action: #selector(cbToggle(_:)), keyEquivalent: "")
        previewItem.target = self
        previewItem.representedObject = "live_preview"
        previewItem.state = cfg.bool("live_preview", default: true) ? .on : .off
        sm.addItem(previewItem)
        toggleItems["live_preview"] = previewItem
        // Cleanup model retention (default ON = keep warm across a dictation
        // burst and idle-unload after 2 min; OFF = drop it after every single
        // dictation, which made each cleanup pay ~1s of reload on the critical
        // path before the paste).
        let warmItem = NSMenuItem(title: "Keep cleanup model warm (faster, more RAM)",
                                  action: #selector(cbToggle(_:)), keyEquivalent: "")
        warmItem.target = self
        warmItem.representedObject = "keep_cleanup_warm"
        warmItem.state = cfg.bool("keep_cleanup_warm", default: true) ? .on : .off
        sm.addItem(warmItem)
        toggleItems["keep_cleanup_warm"] = warmItem
        // Free the ASR model (~2.7 GB CoreML/ANE) after 2 min idle. Default ON.
        let freeIdleItem = NSMenuItem(title: "Free model when idle (lower RAM)",
                                      action: #selector(cbToggle(_:)), keyEquivalent: "")
        freeIdleItem.target = self
        freeIdleItem.representedObject = "free_model_idle"
        freeIdleItem.state = cfg.bool("free_model_idle", default: true) ? .on : .off
        sm.addItem(freeIdleItem)
        toggleItems["free_model_idle"] = freeIdleItem
        settings.submenu = sm
        menu.addItem(settings)

        // Flow swallows the Fn key, but the Globe handler lives below any tap
        // macOS lets an app install, so the only guaranteed cure for the emoji
        // picker is the system setting itself.
        let fnItem = NSMenuItem(title: "Fn key opens emoji picker? Fix in Settings…",
                                action: #selector(cbOpenKeyboardSettings(_:)), keyEquivalent: "")
        fnItem.target = self
        menu.addItem(fnItem)

        let freeItem = NSMenuItem(title: "Free memory (unload models)", action: #selector(cbFreeMemory(_:)), keyEquivalent: "")
        freeItem.target = self
        menu.addItem(freeItem)
        let resetItem = NSMenuItem(title: "Stop / Reset", action: #selector(cbEmergencyReset(_:)), keyEquivalent: "")
        resetItem.target = self
        menu.addItem(resetItem)
        menu.addItem(.separator())

        statusMenuItem = NSMenuItem(title: "Loading model…", action: #selector(cbStatusClick(_:)), keyEquivalent: "")
        statusMenuItem.target = self
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())

        let kofi = NSMenuItem(title: "Support on Ko-fi…", action: #selector(cbKofi(_:)), keyEquivalent: "")
        kofi.target = self
        menu.addItem(kofi)
        menu.addItem(.separator())

        let restart = NSMenuItem(title: "Restart Flow", action: #selector(cbRestart(_:)), keyEquivalent: "")
        restart.target = self
        menu.addItem(restart)
        let quit = NSMenuItem(title: "Quit Flow", action: #selector(cbQuit(_:)), keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
        return menu
    }

    func disabled(_ title: String) -> NSMenuItem {
        let it = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        it.isEnabled = false
        return it
    }

    func rebuildHistoryMenu() {
        guard let store else { return }
        historyMenu.removeAllItems()
        let rows = store.recent(limit: MAX_HISTORY)
        for i in 0..<MAX_HISTORY {
            if i < rows.count {
                let snippet = rows[i].1.count > 52
                    ? String(rows[i].1.prefix(52)) + "…" : rows[i].1
                let it = NSMenuItem(title: "\(i + 1). \u{201C}\(snippet)\u{201D}",
                                    action: #selector(cbPasteFromHistory(_:)), keyEquivalent: "")
                it.target = self
                it.representedObject = rows[i].1
                historyMenu.addItem(it)
            } else {
                historyMenu.addItem(disabled("(empty slot \(i + 1))"))
            }
        }
        historyMenu.addItem(.separator())
        let clear = NSMenuItem(title: "Clear history", action: #selector(cbClearHistory(_:)), keyEquivalent: "")
        clear.target = self
        historyMenu.addItem(clear)

        if let first = rows.first {
            let preview = first.1.count > 52 ? String(first.1.prefix(52)) + "…" : first.1
            lastPasteItem?.title = "Last paste — \u{201C}\(preview)\u{201D}"
        }
        ui.reloadHistory()
    }

    // MARK: - Event tap (Fn hotkey)

    var tapPermissionNotified = false

    func startEventTapThread() {
        Thread.detachNewThread { [self] in
            while true {
                let ok = runEventTap()
                Thread.sleep(forTimeInterval: ok ? 2.0 : 3.0)
                if ok { flowLog("[tap] event tap loop exited, reinstalling") }
            }
        }
    }

    func runEventTap() -> Bool {
        // Prefer an HID-level tap. A session tap sees the Fn key only after the
        // system's Globe handler has looked at it, which is why a quick double
        // press could still pop the emoji picker even though Flow consumes the
        // event. At HID level we sit ahead of that handler. Fall back to the
        // session tap if the system refuses one.
        let mask = CGEventMask(1 << CGEventType.flagsChanged.rawValue)
        let userInfo = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let me = Unmanaged<AppDelegate>.fromOpaque(refcon).takeUnretainedValue()
            return me.handleTapEvent(type: type, event: event)
        }
        var tapLevel = "hid"
        var tapCandidate = CGEvent.tapCreate(
            tap: .cghidEventTap, place: .headInsertEventTap, options: .defaultTap,
            eventsOfInterest: mask, callback: callback, userInfo: userInfo)
        if tapCandidate == nil {
            tapLevel = "session"
            tapCandidate = CGEvent.tapCreate(
                tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                eventsOfInterest: mask, callback: callback, userInfo: userInfo)
        }
        guard let tap = tapCandidate else {
            if !tapPermissionNotified {
                tapPermissionNotified = true
                flowLog("[tap] CGEventTap creation FAILED (permissions?)")
                updateStatusLine("⚠️  Grant Accessibility/Input Monitoring — click to open Settings")
                notify("Flow — Action Required",
                       "System Settings → Privacy & Security → allow Flow in Accessibility/Input Monitoring")
            }
            return false
        }
        if tapPermissionNotified {
            tapPermissionNotified = false
            updateStatusLine(modelReady ? "Parakeet v3 CoreML · FluidAudio · ready" : "Loading model…")
        }
        eventTap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        flowLog("[tap] event tap installed (\(tapLevel) level)")
        CFRunLoopRun()
        return true
    }

    func handleTapEvent(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            // Log it: re-enabling silently made a mid-dictation tap disable
            // indistinguishable from a normal Fn release in the log.
            flowLog("[tap] tap disabled (type=\(type.rawValue)); re-enabling, fnDown=\(fnDown)")
            if let tap = eventTap { CGEvent.tapEnable(tap: tap, enable: true) }
            if fnDown {
                fnDown = false
                hotkeyQueue.async { self.onFnRelease() }
            }
            return Unmanaged.passUnretained(event)
        }
        guard type == .flagsChanged else { return Unmanaged.passUnretained(event) }
        let fnNow = (event.flags.rawValue & HOTKEY_FLAG) != 0
        if cfg.bool("log_key_events", default: false) {
            let src = event.getIntegerValueField(.eventSourceUserData)
            flowLog(String(format: "[tap] flags=0x%llX fn=%@ was=%@ src=%lld",
                           event.flags.rawValue, fnNow ? "1" : "0", fnDown ? "1" : "0", src))
        }
        if fnNow != fnDown {
            fnDown = fnNow
            if fnNow {
                hotkeyQueue.async { self.onFnPress() }
            } else {
                hotkeyQueue.async { self.onFnRelease() }
            }
            // Consume the event: suppresses the native Fn action
            // (emoji picker / macOS dictation).
            return nil
        }
        return Unmanaged.passUnretained(event)
    }

    // MARK: - Press / release logic (port)

    func onFnPress() {
        let now = ProcessInfo.processInfo.systemUptime
        if recorder.isRecording && recorder.isHandsFree {
            // Press during hands-free: stop and transcribe
            stopRecording()
            return
        }
        if let timer = deferredStopTimer {
            // Second press within the double-press window: hands-free promotion
            timer.cancel()
            deferredStopTimer = nil
            recorder.promoteToHandsFree()
            setState("rec_hf")
            flowLog("[rec] hands-free (double press, batch transcribe on stop)")
            return
        }
        guard modelReady else {
            notify("Still loading", "Model is loading — try again in a moment.")
            return
        }
        lastFn = now
        startRecording()
    }

    func onFnRelease() {
        guard recorder.isRecording, !recorder.isHandsFree else { return }
        let pressDur = ProcessInfo.processInfo.systemUptime - lastFn
        if pressDur < DOUBLE_PRESS_SEC {
            let delay = (DOUBLE_PRESS_SEC - pressDur) + 0.05
            let work = DispatchWorkItem { [weak self] in
                self?.deferredStopTimer = nil
                self?.stopRecording()
            }
            deferredStopTimer = work
            hotkeyQueue.asyncAfter(deadline: .now() + delay, execute: work)
        } else {
            stopRecording()
        }
    }

    func startRecording() {
        ipcRecordingContext = nil
        do {
            try beginRecording()
        } catch {
            flowLog("[rec] start failed: \(error)")
            notify("Transcription error", "\(error.localizedDescription)")
            setState("idle")
        }
    }

    func beginRecording() throws {
        // Never restart on top of a live capture: recorder.start() clears the
        // buffers, so a stray second press would throw away everything already
        // spoken and the dictation would silently produce nothing.
        guard !recorder.isRecording else {
            flowLog("[rec] start ignored: already recording")
            return
        }
        preloadASR()
        try recorder.start()
        firstFragmentPasted = false
        setState("rec")
        playSound("/System/Library/Sounds/Pop.aiff")
        flowLog("[rec] recording started")
        startLivePreview()
    }

    @discardableResult
    func stopRecording() -> Bool {
        stopLivePreview()
        let ipcContext = ipcRecordingContext
        ipcRecordingContext = nil
        let capturedAudio = recorder.stop()
        playSound("/System/Library/Sounds/Tink.aiff")
        let audio = trimTrailingSilence(capturedAudio)
        let activity = voiceActivity(in: audio)
        let capturedAudioS = Double(capturedAudio.count) / Double(FlowAudio.sampleRate)
        let audioS = Double(audio.count) / Double(FlowAudio.sampleRate)
        flowLog(String(format: "[rec] stopped: %.2fs captured, %.2fs trimmed, voiced=%d/%d",
                       capturedAudioS, audioS, activity.voicedWindows, activity.totalWindows))

        // Accidental sub-second taps are classified FIRST and stay silent, as
        // they always have. Only a capture the user plausibly meant reaches the
        // voice gate below, so the notification there never fires on a slip.
        if capturedAudio.count <= FlowAudio.sampleRate {  // <= 1.0 s: discard
            discardRecording(ipcContext, reason: "audio_too_short")
            return false
        }

        // This runs for both push-to-talk and hands-free. It must happen before
        // the worker is queued so silence can never reach ASR, partial/final,
        // or cleanup IPC events.
        guard activity.hasVoice else {
            flowLog(String(format: "[rec] discarded silent audio (peak RMS %.4f)", Double(activity.peakRMS)))
            // Say so. Silently dropping a real dictation is indistinguishable
            // from a failed paste, and the user just re-dictates blind.
            if ipcContext == nil {
                playSound("/System/Library/Sounds/Basso.aiff")
                notify("Nothing recorded", "No speech detected — the capture was discarded.")
            }
            discardRecording(ipcContext, reason: "silent_audio")
            return false
        }

        setState("proc")
        if let correlationID = ipcContext?.correlationID {
            emitIPCStatus(correlationID, "processing")
        }
        enqueue(audio: audio,
                correlationID: ipcContext?.correlationID,
                pasteOutput: ipcContext?.pasteOutput ?? true)
        return true
    }

    /// End a capture before ASR while still giving IPC clients a terminal,
    /// machine-readable reason. Local hotkey captures simply return to idle.
    func discardRecording(_ ipcContext: IPCRecordingContext?, reason: String) {
        setState("idle")
        guard let correlationID = ipcContext?.correlationID else { return }
        emitIPC(.init(type: .error, correlationId: correlationID, message: reason))
        emitIPCStatus(correlationID, "idle")
        finishIPCCorrelation(correlationID)
    }

    func enqueueUtterance(_ audio: [Float]) {
        enqueue(audio: audio, keepRecordingState: true)
    }

    var idleUnloadWork: DispatchWorkItem? = nil

    /// When the cleanup model is kept warm, unload it after 2 min of no
    /// dictation so RAM isn't held forever. Reset on every new dictation.
    func scheduleIdleUnload() {
        idleUnloadWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            if self.pendingItems <= 0 && !self.recorder.isRecording {
                self.cleanupModel.unload()
                flowLog("[ai-unload] idle unload of cleanup LLM (kept warm)")
            }
        }
        idleUnloadWork = work
        DispatchQueue.global().asyncAfter(deadline: .now() + 120, execute: work)
    }

    var asrIdleUnloadWork: DispatchWorkItem? = nil
    // Seconds of no dictation before the ASR model (the ~2.7 GB CoreML/ANE
    // footprint) is released. The reload is NOT ~0.1s as originally assumed:
    // measured on this machine it costs 1s warm and 11-35s cold, and it is paid
    // after the user stops speaking, with the HUD sitting on "Transcribing…".
    // At 120s every normal pause between dictations bought that stall, so the
    // window is 15 min — long enough that a working session never pays it,
    // short enough that Flow still gives the RAM back when actually idle.
    let asrIdleUnloadSeconds: Double =
        Double(ProcessInfo.processInfo.environment["FLOW_ASR_IDLE"] ?? "") ?? 900

    // MARK: - Live preview while recording

    var livePreviewRunning = false
    /// How much trailing audio each preview pass transcribes. Bounded on
    /// purpose: re-running the whole session would grow without limit and a
    /// 7-minute dictation would spend more time previewing than recording.
    let livePreviewTailSeconds = 18.0
    let livePreviewIntervalSeconds = 2.5

    func startLivePreview() {
        guard cfg.bool("live_preview", default: true), !livePreviewRunning else { return }
        livePreviewRunning = true
        hud?.clearPreview()
        Task.detached(priority: .utility) { [self] in
            let tail = Int(livePreviewTailSeconds * Double(FlowAudio.sampleRate))
            while livePreviewRunning && recorder.isRecording {
                let session = recorder.sessionAudio()
                if session.count >= FlowAudio.sampleRate {
                    let slice = session.count > tail ? Array(session.suffix(tail)) : session
                    if voiceActivity(in: slice).hasVoice,
                       let text = try? await transcriber.transcribe(slice, language: nil).text,
                       !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       livePreviewRunning, recorder.isRecording {
                        let shown = String(text.suffix(160))
                        hud?.setPreview(shown, state: recorder.isHandsFree ? "rec_hf" : "rec")
                        await MainActor.run { self.ui.preview = shown }
                    }
                }
                try? await Task.sleep(nanoseconds: UInt64(livePreviewIntervalSeconds * 1_000_000_000))
            }
        }
    }

    func stopLivePreview() {
        livePreviewRunning = false
        hud?.clearPreview()
        DispatchQueue.main.async { self.ui.preview = "" }
    }

    // MARK: - Burst tracking

    /// True when this dictation is aimed at the same field as the previous one,
    /// close enough in time, and that text is still sitting untouched at the end
    /// of the field. Any of those failing starts a fresh burst, so a stale burst
    /// can never make Flow rewrite text the user has since edited.
    func burstContinues(pid: pid_t) -> Bool {
        guard cfg.bool("merge_bursts", default: true), let b = burst else { return false }
        guard pid != 0, pid == b.pid else { return false }
        guard -b.lastActivity.timeIntervalSinceNow < burstWindowSeconds else { return false }
        guard !b.textInField.isEmpty else { return false }
        return paster.focusedTextEndsWith(b.textInField)
    }

    func noteBurstPaste(text: String, pid: pid_t) {
        guard pid != 0, !text.isEmpty else { burst = nil; return }
        burst = BurstState(pid: pid, textInField: text, lastActivity: Date())
    }

    /// Release the ASR model after idle so Flow doesn't hold ~2.7 GB when
    /// unused. Disabled by turning off "Free model when idle" in Settings.
    func scheduleAsrIdleUnload() {
        asrIdleUnloadWork?.cancel()
        guard cfg.bool("free_model_idle", default: true) else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            if self.pendingItems <= 0 && !self.recorder.isRecording {
                Task.detached { [self] in
                    await self.transcriber.unload()
                    flowLog("[asr-unload] idle unload of ASR model (RAM freed)")
                }
            }
        }
        asrIdleUnloadWork = work
        DispatchQueue.global().asyncAfter(deadline: .now() + asrIdleUnloadSeconds, execute: work)
    }

    /// On Fn press: cancel the pending unload and warm the model back up in the
    /// background so it's ready by the time the user stops talking.
    func preloadASR() {
        if cfg.aiBackend == "local" {
            DispatchQueue.global(qos: .userInitiated).async { [self] in cleanupModel.warmUp() }
        }
        asrIdleUnloadWork?.cancel()
        asrIdleUnloadWork = nil
        if !transcriber.isLoaded {
            Task.detached { [self] in
                do { try await self.transcriber.ensureLoaded() }
                catch { flowLog("[asr] preload failed: \(error)") }
            }
        }
    }

    func enqueue(audio: [Float], keepRecordingState: Bool = false,
                 correlationID: String? = nil, pasteOutput: Bool = true) {
        pendingItems += 1
        workQueue.async { [self] in
            processAudio(audio, correlationID: correlationID, pasteOutput: pasteOutput)
            pendingItems -= 1
            if let correlationID {
                if !isIPCCancelled(correlationID) { emitIPCStatus(correlationID, "idle") }
                finishIPCCorrelation(correlationID)
            }
            if pendingItems <= 0 && !recorder.isRecording {
                if cfg.bool("keep_cleanup_warm", default: true) {
                    // Keep the model resident, unload only after idle.
                    scheduleIdleUnload()
                } else {
                    // RAM back immediately: drop the cleanup LLM after the burst.
                    cleanupModel.unload()
                }
                // Release the ASR model too after the idle grace period.
                scheduleAsrIdleUnload()
                setState("idle")
            } else if keepRecordingState && recorder.isRecording {
                setState(recorder.isHandsFree ? "rec_hf" : "rec")
            }
        }
    }

    // MARK: - Pipeline (port of the worker)

    func processAudio(_ rawAudio: [Float], correlationID: String? = nil,
                      pasteOutput: Bool = true) {
        let tStart = Date()
        var timing: [String: Double] = [:]
        let audio = trimTrailingSilence(rawAudio)
        let activity = voiceActivity(in: audio)
        guard activity.hasVoice else {
            flowLog(String(format: "[worker] skipped silent audio (peak RMS %.4f)", Double(activity.peakRMS)))
            if let correlationID, !isIPCCancelled(correlationID) {
                emitIPC(.init(type: .error, correlationId: correlationID,
                              message: "silent_audio"))
            }
            return
        }
        let audioS = Double(audio.count) / Double(FlowAudio.sampleRate)
        timing["audio_s"] = audioS

        // 1. Transcribe (full auto: no language hint, model self-detects)
        // If the model was idle-unloaded, the reload happens inside transcribe()
        // and can take seconds. Say so instead of showing "Transcribing…" for
        // the whole wait, which is what read as "it stays loading a long time".
        // (not during hands-free: there the pill must stay on "Hands-free")
        let showsProgress = !recorder.isRecording
        if showsProgress && !transcriber.isLoaded { setState("load") }
        var raw = ""
        let sem = DispatchSemaphore(value: 0)
        var trError: Error? = nil
        Task.detached { [self] in
            do {
                let r = try await transcriber.transcribe(audio, language: nil)
                raw = r.text
                timing["transcribe_s"] = r.transcribeS
                timing["load_s"] = r.loadS
            } catch {
                trError = error
            }
            sem.signal()
        }
        sem.wait()
        if showsProgress { setState("proc") }
        if let e = trError {
            flowLog("[worker] transcription failed: \(e)")
            notify("Transcription error", "\(e.localizedDescription)")
            if let correlationID {
                emitIPC(.init(type: .error, correlationId: correlationID,
                              message: e.localizedDescription))
            }
            return
        }
        if let correlationID, isIPCCancelled(correlationID) { return }
        flowLog("[transcribe] raw: \(raw)")
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let correlationID {
                emitIPC(.init(type: .error, correlationId: correlationID,
                              message: "empty_transcription"))
            }
            return
        }
        if let correlationID {
            emitIPC(.init(type: .partial, correlationId: correlationID, text: raw))
        }

        // Detect the spoken language from the transcript so cleanup uses the
        // right few-shot + language lock (Flow figures the language out itself).
        let language = FlowLanguage.detect(raw)
        if let language { flowLog("[lang] detected: \(language)") }

        // 2. Filters
        var text = raw
        text = stripStockPhrases(text)
        text = stripTrailingRepeat(text)
        text = stripTrailingEcho(text)
        text = stripRepeatLoop(text)
        if isHallucination(text, expectedLanguage: language) {
            flowLog("[worker] hallucination discarded: \(text)")
            if let correlationID {
                emitIPC(.init(type: .error, correlationId: correlationID,
                              message: "hallucination_discarded"))
            }
            return
        }

        // 3. Deterministic processing
        text = processText(text,
                           verbalCommands: cfg.bool("verbal_commands", default: true),
                           removeFillers: cfg.removeFillers)
        if text == DELETE_SENTINEL {
            if pasteOutput { paster.undo() }
            if let correlationID {
                emitIPC(.init(type: .final, correlationId: correlationID, text: ""))
                emitIPC(.init(type: .cleanup, correlationId: correlationID, text: ""))
            }
            return
        }
        text = correctDictionaryTerms(text, terms: cfg.userDictionary)
        if text.isEmpty { return }
        if let correlationID {
            emitIPC(.init(type: .final, correlationId: correlationID, text: text))
        }
        if let correlationID, isIPCCancelled(correlationID) { return }

        // 4. AI cleanup
        let ctx = paster.captureContext(includeText: cfg.bool("verify_paste", default: true))

        // Is this a continuation of the run of dictations already in the field?
        // If so the cleanup gets the whole burst, so it can punctuate across the
        // seam between fragments instead of seeing each one in isolation.
        let continuing = pasteOutput && burstContinues(pid: paster.focusedTargetPid())
        let prefix = (continuing || firstFragmentPasted) ? " " : ""
        let priorInField = continuing ? (burst?.textInField ?? "") : ""
        let cleanupInput = priorInField.isEmpty ? text : priorInField + " " + text

        // Instant paste: cleanup is a multi-second edit that, measured over 185
        // real runs, mostly inserts commas and fixes capitals. Waiting for it
        // before showing anything is what made Flow feel slow. Put the draft on
        // screen straight away and polish it in place afterwards.
        let instantPaste = pasteOutput
            && cfg.bool("instant_paste", default: true)
            && ai.wouldRunModel(cleanupInput)
        var draftPasted: String? = nil
        var draftResult: PasteResult? = nil
        if instantPaste {
            draftResult = paster.paste(prefix + text, contextBefore: ctx)
            // Keep the draft even when verification failed. An unconfirmed
            // paste is usually a paste that landed but that the target's
            // accessibility tree published late, and pasting a second time
            // would duplicate the whole dictation in the field.
            if draftResult?.ok != false { firstFragmentPasted = true }
            draftPasted = prefix + text
        }

        var aiUsed = 0
        var aiKept = 0
        var cleaned = cleanupInput
        if ai.isEnabled() {
            // Text is already on screen: say what is actually happening rather
            // than leaving the pill on "Transcribing…".
            if showsProgress && draftPasted != nil { setState("ai") }
            let t0 = Date()
            cleaned = ai.clean(cleanupInput, appBundle: ctx.bundleId, language: language)
            timing["cleanup_s"] = -t0.timeIntervalSinceNow
            aiUsed = 1
            aiKept = (cleaned != cleanupInput) ? 1 : 0
        }
        if let correlationID, isIPCCancelled(correlationID) { return }

        // What belongs in the field for the whole burst, and what this dictation
        // contributed on its own (history and IPC stay per-dictation).
        let burstText = cleaned
        let finalText = priorInField.isEmpty
            ? cleaned
            : Self.tailContribution(of: cleaned, after: priorInField.count, fallback: text)
        if let correlationID, isIPCCancelled(correlationID) { return }
        if let correlationID {
            emitIPC(.init(type: .cleanup, correlationId: correlationID, text: finalText))
        }

        // 6. Paste (or upgrade what is already on screen)
        var pasteResult: PasteResult? = draftResult
        // What the burst occupies in the field once this dictation is done.
        var textNowInField = priorInField.isEmpty ? text : priorInField + " " + text
        if let draft = draftPasted {
            // On a continuation the whole burst is rewritten, so the cleaned
            // punctuation spans the seam between fragments.
            let oldTail = priorInField.isEmpty ? draft : priorInField + draft
            let newTail = priorInField.isEmpty ? prefix + burstText : burstText
            let outcome = paster.replaceDraft(oldTail, with: newTail,
                                              expectedPid: draftResult?.targetPid ?? 0)
            if outcome != .identical {
                flowLog("[paste] polish: \(outcome.rawValue)\(continuing ? " (burst)" : "")")
            }
            if outcome == .replaced || outcome == .identical {
                pasteResult = PasteResult(ok: true, method: "instant+polish", error: nil)
                textNowInField = burstText
            }
            pasteResult?.pasteS = draftResult?.pasteS ?? 0
            pasteResult?.targetPid = draftResult?.targetPid ?? 0
        } else if pasteOutput {
            pasteResult = paster.paste(prefix + finalText, contextBefore: ctx)
            // Only claim a fragment landed if it did, otherwise the next
            // fragment gets a stray leading space.
            if pasteResult?.ok != false { firstFragmentPasted = true }
            textNowInField = priorInField.isEmpty ? finalText : priorInField + " " + finalText
        }
        if pasteOutput, pasteResult?.ok != false {
            noteBurstPaste(text: textNowInField, pid: pasteResult?.targetPid ?? 0)
        } else {
            burst = nil
        }
        if pasteResult?.ok == false {
            // Recovery copy is the finished text, not the draft that was on the
            // clipboard when the draft paste failed.
            if draftPasted != nil { paster.stageOnClipboard(finalText) }
            // The notification banner is not always delivered, so make the
            // failure audible too: the transcript is on the clipboard and the
            // user needs to know to press ⌘V rather than re-dictate.
            playSound("/System/Library/Sounds/Basso.aiff")
            notify("Paste not confirmed", "Text is on the clipboard — press ⌘V to paste it.")
        }

        // 7. History + timing
        var rec = HistoryRecord(createdAt: Self.timestamp(), text: finalText)
        rec.rawText = raw
        rec.language = language
        rec.appBundle = ctx.bundleId
        rec.appName = ctx.appName
        rec.url = ctx.url
        rec.focusedRole = ctx.focusedRole
        rec.focusedText = ctx.focusedText.map { String($0.prefix(1200)) }
        rec.audioS = audioS
        rec.transcribeS = timing["transcribe_s"]
        rec.cleanupS = timing["cleanup_s"]
        rec.pasteS = pasteResult?.pasteS
        rec.totalS = -tStart.timeIntervalSinceNow
        rec.charsIn = raw.count
        rec.charsOut = finalText.count
        rec.backend = "fluidaudio-coreml"
        rec.model = transcriber.modelLabel
        rec.aiBackend = ai.backend
        rec.aiUsed = aiUsed
        rec.aiKept = aiKept
        rec.snippetCount = 0
        rec.pasteOk = pasteResult?.ok.map { $0 ? 1 : 0 }
        rec.pasteMethod = pasteResult?.method ?? (pasteOutput ? nil : "ipc")
        rec.pasteError = pasteResult?.error
        store?.insert(rec)
        DispatchQueue.main.async { self.rebuildHistoryMenu() }

        emitTiming(rec, loadS: timing["load_s"] ?? 0)
    }

    /// The part of a cleaned burst contributed by its last fragment, for the
    /// history record only. Cleanup shifts lengths by a few characters, so the
    /// cut is snapped to a word boundary and abandoned if it looks implausible.
    static func tailContribution(of cleaned: String, after priorLength: Int,
                                 fallback: String) -> String {
        guard priorLength > 0, priorLength < cleaned.count else { return fallback }
        var idx = cleaned.index(cleaned.startIndex, offsetBy: priorLength)
        while idx < cleaned.endIndex, cleaned[idx] != " " {
            idx = cleaned.index(after: idx)
        }
        let tail = String(cleaned[idx...]).trimmingCharacters(in: .whitespaces)
        guard tail.count >= fallback.count / 2 else { return fallback }
        return tail
    }

    static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f.string(from: Date())
    }

    /// `loadS` is the ASR model reload, which is paid inside transcribe() but is
    /// not part of transcribeS. Reporting it separately keeps `total` fully
    /// attributed — an unattributed gap here is what hid the reload stalls.
    func emitTiming(_ r: HistoryRecord, loadS: Double = 0) {
        let line = String(
            format: "[timing] audio=%.3fs load=%.3fs transcribe=%.3fs cleanup=%.3fs paste=%.3fs total=%.3fs in→out=%d→%d ai=%@ paste=%@",
            r.audioS ?? 0, loadS, r.transcribeS ?? 0, r.cleanupS ?? 0, r.pasteS ?? 0, r.totalS ?? 0,
            r.charsIn ?? 0, r.charsOut ?? 0,
            (r.aiUsed == 1 ? (r.aiKept == 1 ? "✓" : "∅") : "-"),
            "\(r.pasteOk.map { $0 == 1 ? "ok" : "fail" } ?? "?"):\(r.pasteMethod ?? "?")")
        flowLog(line)
        let csvPath = FlowConfig.flowDir.appendingPathComponent("timings.csv")
        let header = "timestamp,audio_s,transcribe_s,cleanup_s,paste_s,total_s,chars_in,chars_out,language,backend,model,ai_backend,ai_used,ai_kept\n"
        let row = String(format: "%@,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%@,%@,%@,%@,%d,%d\n",
                         r.createdAt, r.audioS ?? 0, r.transcribeS ?? 0, r.cleanupS ?? 0,
                         r.pasteS ?? 0, r.totalS ?? 0, r.charsIn ?? 0, r.charsOut ?? 0,
                         r.language ?? "", r.backend ?? "", r.model ?? "",
                         r.aiBackend ?? "", r.aiUsed ?? 0, r.aiKept ?? 0)
        if !FileManager.default.fileExists(atPath: csvPath.path) {
            try? header.write(to: csvPath, atomically: true, encoding: .utf8)
        }
        if let fh = try? FileHandle(forWritingTo: csvPath) {
            fh.seekToEndOfFile()
            fh.write(row.data(using: .utf8)!)
            try? fh.close()
        }
    }

    // MARK: - Sounds / notifications

    func playSound(_ path: String) {
        guard cfg.soundFeedback else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        p.arguments = [path]
        try? p.run()
    }

    func notify(_ title: String, _ message: String) {
        DispatchQueue.main.async {
            let n = NSUserNotification()
            n.title = APP_NAME
            n.subtitle = title
            n.informativeText = message
            NSUserNotificationCenter.default.deliver(n)
        }
        flowLog("[notify] \(title): \(message)")
    }

    // MARK: - Stable IPC v1 (Darkadyan)

    func setupStableIPC() {
        let server = FlowIPCServer { [weak self] request in
            guard let self else { return .failure(id: request.id, "app_unavailable") }
            return self.handleIPCRequest(request)
        }
        do {
            try server.start()
            ipcServer = server
            flowLog("[ipc-v1] listening at \(server.socketURL.path)")
        } catch {
            flowLog("[ipc-v1] start failed: \(error.localizedDescription)")
        }
    }

    func handleIPCRequest(_ request: FlowIPCRequest) -> FlowIPCResponse {
        switch request.method {
        case .health:
            return .success(id: request.id, ipcStatusPayload(includeHealth: true))

        case .status:
            return .success(id: request.id, ipcStatusPayload(includeHealth: false))

        case .start:
            let suppliedCorrelationID = request.params["correlationId"]?.stringValue?.trimmingCharacters(
                in: .whitespacesAndNewlines
            )
            let correlationID = (suppliedCorrelationID?.isEmpty == false)
                ? suppliedCorrelationID!
                : request.id
            let pasteOutput = request.params["paste"]?.boolValue ?? false
            var response = FlowIPCResponse.failure(id: request.id, "start_failed")
            onMain {
                guard self.modelReady else {
                    response = .failure(id: request.id, "model_not_ready")
                    return
                }
                guard !self.recorder.isRecording, self.pendingItems <= 0 else {
                    response = .failure(id: request.id, "busy")
                    return
                }
                self.registerIPCCorrelation(correlationID)
                self.ipcRecordingContext = IPCRecordingContext(
                    correlationID: correlationID,
                    pasteOutput: pasteOutput
                )
                do {
                    try self.beginRecording()
                    self.emitIPCStatus(correlationID, "recording")
                    response = .success(id: request.id, [
                        "correlationId": .string(correlationID),
                        "state": .string("recording"),
                    ])
                } catch {
                    self.ipcRecordingContext = nil
                    self.finishIPCCorrelation(correlationID)
                    response = .failure(id: request.id, error.localizedDescription)
                }
            }
            return response

        case .stop:
            let requestedID = request.params["correlationId"]?.stringValue
            var response = FlowIPCResponse.failure(id: request.id, "not_recording")
            onMain {
                guard self.recorder.isRecording else { return }
                let correlationID = self.ipcRecordingContext?.correlationID
                guard requestedID == nil || requestedID == correlationID else {
                    response = .failure(id: request.id, "correlation_mismatch")
                    return
                }
                let processing = self.stopRecording()
                response = .success(id: request.id, [
                    "correlationId": correlationID.map(FlowIPCValue.string) ?? .null,
                    "state": .string(processing ? "processing" : "idle"),
                ])
            }
            return response

        case .cancel:
            let requestedID = request.params["correlationId"]?.stringValue
            var response = FlowIPCResponse.failure(id: request.id, "nothing_to_cancel")
            onMain {
                let recordingID = self.ipcRecordingContext?.correlationID
                let targetID = requestedID ?? recordingID ?? self.latestIPCCorrelation()
                guard let targetID, self.cancelIPCCorrelation(targetID) else { return }

                if self.recorder.isRecording,
                   recordingID == targetID || requestedID == nil {
                    _ = self.recorder.stop()
                    self.ipcRecordingContext = nil
                    self.deferredStopTimer?.cancel()
                    self.deferredStopTimer = nil
                    self.firstFragmentPasted = false
                    self.setState(self.modelReady ? "idle" : "loading")
                }
                self.emitIPCStatus(targetID, "cancelled")
                response = .success(id: request.id, [
                    "correlationId": .string(targetID),
                    "state": .string("cancelled"),
                ])
            }
            return response
        }
    }

    func ipcStatusPayload(includeHealth: Bool) -> [String: FlowIPCValue] {
        onMain {
            var payload: [String: FlowIPCValue] = [
                "protocolVersion": .int(FlowIPCProtocol.version),
                "bundleIdentifier": .string(FlowIPCProtocol.bundleIdentifier),
                "socketPath": .string(FlowIPCProtocol.socketURL.path),
                "state": .string(self.state),
                "modelReady": .bool(self.modelReady),
                "recording": .bool(self.recorder.isRecording),
                "pendingItems": .int(self.pendingItems),
            ]
            if includeHealth {
                payload["available"] = .bool(true)
                payload["pid"] = .int(Int(getpid()))
            }
            return payload
        }
    }

    func emitIPC(_ event: FlowIPCEvent) {
        ipcServer?.emit(event)
    }

    func emitIPCStatus(_ correlationID: String, _ state: String) {
        emitIPC(.init(type: .status, correlationId: correlationID, state: state))
    }

    func registerIPCCorrelation(_ correlationID: String) {
        ipcSessionLock.lock()
        ipcCancelledCorrelations.remove(correlationID)
        ipcLatestCorrelationID = correlationID
        ipcSessionLock.unlock()
    }

    func cancelIPCCorrelation(_ correlationID: String) -> Bool {
        ipcSessionLock.lock()
        defer { ipcSessionLock.unlock() }
        guard ipcLatestCorrelationID == correlationID else { return false }
        ipcCancelledCorrelations.insert(correlationID)
        return true
    }

    func isIPCCancelled(_ correlationID: String) -> Bool {
        ipcSessionLock.lock()
        defer { ipcSessionLock.unlock() }
        return ipcCancelledCorrelations.contains(correlationID)
    }

    func latestIPCCorrelation() -> String? {
        ipcSessionLock.lock()
        defer { ipcSessionLock.unlock() }
        return ipcLatestCorrelationID
    }

    func finishIPCCorrelation(_ correlationID: String) {
        ipcSessionLock.lock()
        ipcCancelledCorrelations.remove(correlationID)
        if ipcLatestCorrelationID == correlationID { ipcLatestCorrelationID = nil }
        ipcSessionLock.unlock()
    }

    func onMain<T>(_ body: () -> T) -> T {
        if Thread.isMainThread { return body() }
        return DispatchQueue.main.sync(execute: body)
    }

    // MARK: - Debug IPC (E2E test hook)

    func setupDebugIPC() {
        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name("com.shaun.flowswift.debug"),
            object: nil, queue: nil
        ) { [weak self] note in
            guard let self else { return }
            let action = (note.userInfo?["action"] as? String) ?? note.object as? String ?? ""
            flowLog("[debug-ipc] action=\(action)")
            switch action {
            case "start": self.hotkeyQueue.async { self.onFnPress() }
            case "stop": self.hotkeyQueue.async {
                if self.recorder.isRecording {
                    if self.recorder.isHandsFree {
                        self.stopRecording()
                    } else {
                        self.onFnRelease()
                    }
                }
            }
            case let a where a.hasPrefix("hud_demo:"):
                // Mostra l'HUD in uno stato dato senza toccare microfono e
                // pipeline: serve a verificare la UI senza rubare una dettatura.
                let raw = String(a.dropFirst("hud_demo:".count))
                DispatchQueue.main.async {
                    if raw == "off" { self.hud?.hide() }
                    else { self.hud?.show(state: raw) }
                }
            case let a where a.hasPrefix("hud_preview:"):
                let text = String(a.dropFirst("hud_preview:".count))
                self.hud?.setPreview(text, state: "rec")
            case let a where a.hasPrefix("pipeline_file:"):
                // Full dictation pipeline on a wav, paste included, so burst
                // merging and the two-phase paste can be exercised for real.
                let path = String(a.dropFirst("pipeline_file:".count))
                self.preloadASR()
                DispatchQueue.global().async {
                    guard let samples = try? WavIO.loadAsFlowSamples(URL(fileURLWithPath: path)) else {
                        flowLog("[debug-ipc] pipeline_file: cannot read \(path)")
                        return
                    }
                    self.enqueue(audio: samples, pasteOutput: true)
                }
            case let a where a.hasPrefix("paste_test:"):
                // Exercise the two-phase paste against a real window:
                // "paste_test:<draft>|<polished>".
                let payload = String(a.dropFirst("paste_test:".count))
                let parts = payload.components(separatedBy: "|")
                guard parts.count == 2 else { break }
                DispatchQueue.global().async {
                    let ctx = self.paster.captureContext(includeText: true)
                    flowLog("[debug-ipc] target app=\(ctx.appName ?? "?") role=\(ctx.focusedRole ?? "?")")
                    let r = self.paster.paste(parts[0], contextBefore: ctx)
                    flowLog("[debug-ipc] draft paste -> \(r.method) pid=\(r.targetPid)")
                    Thread.sleep(forTimeInterval: 1.0)
                    let mid = self.paster.captureContext(includeText: true)
                    flowLog("[debug-ipc] field now: \(String((mid.focusedText ?? "<nil>").suffix(80)).debugDescription)")
                    let outcome = self.paster.replaceDraft(parts[0], with: parts[1],
                                                           expectedPid: r.targetPid)
                    flowLog("[debug-ipc] polish -> \(outcome.rawValue)")
                }
            case "caret":
                flowLog("[debug-ipc] \(self.paster.describeCaret())")
            case "status":
                flowLog("[debug-ipc] state=\(self.state) modelReady=\(self.modelReady) recording=\(self.recorder.isRecording)")
            case let a where a.hasPrefix("transcribe_file:"):
                // Test hook: run inference on a wav to exercise the real ANE
                // allocation, then let the idle-unload path free it.
                let path = String(a.dropFirst("transcribe_file:".count))
                self.preloadASR()
                Task.detached { [self] in
                    do {
                        let samples = try WavIO.loadAsFlowSamples(URL(fileURLWithPath: path))
                        let r = try await self.transcriber.transcribe(samples, language: nil)
                        // Also exercise the cleanup LLM so the test reproduces
                        // the full memory profile (ASR ANE + Qwen MLX Metal).
                        let cleaned = self.ai.clean(r.text, language: FlowLanguage.detect(r.text), force: true)
                        flowLog("[debug-ipc] transcribe_file -> \(cleaned)")
                    } catch { flowLog("[debug-ipc] transcribe_file failed: \(error)") }
                    await MainActor.run {
                        self.cleanupModel.unload()
                        self.scheduleAsrIdleUnload()
                    }
                }
            default: break
            }
        }
    }

    // MARK: - Callbacks

    @objc func cbSetAiBackend(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        cfg.set("ai_backend", id)
        for it in aiBackendMenu.items { it.state = (it == sender) ? .on : .off }
    }

    @objc func cbSetAiTone(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        cfg.set("ai_tone", id)
        for it in aiToneMenu.items { it.state = (it == sender) ? .on : .off }
    }

    @objc func cbToggle(_ sender: NSMenuItem) {
        guard let key = sender.representedObject as? String else { return }
        let newVal = !(sender.state == .on)
        setFlag(key, newVal)
        sender.state = newVal ? .on : .off
        ui.reloadConfig()
    }

    @objc func cbPasteLast(_ sender: NSMenuItem) {
        guard let rows = store?.recent(limit: 1), let first = rows.first else { return }
        _ = paster.paste(first.1)
    }

    @objc func cbPasteFromHistory(_ sender: NSMenuItem) {
        guard let text = sender.representedObject as? String else { return }
        _ = paster.paste(text)
    }

    @objc func cbClearHistory(_ sender: NSMenuItem) {
        store?.clearHistory()
        DispatchQueue.main.async {
            self.rebuildHistoryMenu()
            self.lastPasteItem.title = "Last paste — none yet"
        }
    }

    @objc func cbEditDictionary(_ sender: NSMenuItem) {
        let current = cfg.userDictionary.joined(separator: "\n")
        promptMultiline(title: "User Dictionary",
                        message: "Words and names the transcriber should recognize better.\nOne per line. Examples: brand names, contacts, technical terms.",
                        text: current) { [self] newText in
            guard let newText else { return }
            var seen = Set<String>()
            var terms: [String] = []
            for piece in newText.split(whereSeparator: { "\n,;".contains($0) }) {
                let t = piece.trimmingCharacters(in: .whitespaces)
                if !t.isEmpty && !seen.contains(t.lowercased()) {
                    seen.insert(t.lowercased())
                    terms.append(t)
                }
            }
            cfg.set("user_dictionary", terms)
            notify("Dictionary updated", "\(terms.count) terms — applied on next dictation.")
        }
    }

    /// Mine the user's own dictation history for words that look like recurring
    /// names or jargon and are not in the dictionary yet, then let them approve
    /// the list. Beats noticing mis-transcriptions by ear one at a time.
    @objc func cbSuggestDictionary(_ sender: NSMenuItem) {
        guard let store else { return }
        let existing = cfg.userDictionary
        DispatchQueue.global().async { [self] in
            let transcripts = store.recentTranscripts(limit: 1500)
            let suggestions = suggestDictionaryTerms(from: transcripts, existing: existing)
            DispatchQueue.main.async {
                guard !suggestions.isEmpty else {
                    self.notify("Dictionary", "No new recurring terms found in your history.")
                    return
                }
                let body = suggestions.map { "\($0.term)" }.joined(separator: "\n")
                let counts = suggestions.map { "\($0.term) (\($0.count)x)" }.joined(separator: ", ")
                flowLog("[dict] suggestions: \(counts)")
                self.promptMultiline(
                    title: "Suggested Dictionary Terms",
                    message: "Recurring capitalised words from your last \(transcripts.count) dictations "
                        + "that are not in the dictionary yet.\nDelete the ones you don't want, then OK to add them.",
                    text: body
                ) { [self] approved in
                    guard let approved else { return }
                    var seen = Set(existing.map { $0.lowercased() })
                    var terms = existing
                    for line in approved.split(separator: "\n") {
                        let t = line.trimmingCharacters(in: .whitespaces)
                        if !t.isEmpty && !seen.contains(t.lowercased()) {
                            seen.insert(t.lowercased())
                            terms.append(t)
                        }
                    }
                    cfg.set("user_dictionary", terms)
                    notify("Dictionary updated", "\(terms.count) terms — applied on next dictation.")
                }
            }
        }
    }

    func promptMultiline(title: String, message: String, text: String,
                         completion: @escaping (String?) -> Void) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = message
            alert.addButton(withTitle: "Save")
            alert.addButton(withTitle: "Cancel")
            let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 380, height: 180))
            let tv = NSTextView(frame: scroll.bounds)
            tv.string = text
            tv.isRichText = false
            tv.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
            scroll.documentView = tv
            scroll.hasVerticalScroller = true
            alert.accessoryView = scroll
            NSApp.activate(ignoringOtherApps: true)
            let resp = alert.runModal()
            completion(resp == .alertFirstButtonReturn ? tv.string : nil)
        }
    }

    @objc func cbSetCurrentAppTone(_ sender: NSMenuItem) {
        guard let app = NSWorkspace.shared.frontmostApplication,
              let bundle = app.bundleIdentifier else {
            notify("No app detected", "Could not identify the frontmost app.")
            return
        }
        let appName = app.localizedName ?? bundle
        DispatchQueue.main.async { [self] in
            let alert = NSAlert()
            alert.messageText = "Current App Tone"
            alert.informativeText = "Cleanup tone for \(appName)"
            alert.addButton(withTitle: "Save")
            alert.addButton(withTitle: "Cancel")
            let popup = NSPopUpButton(frame: NSRect(x: 0, y: 0, width: 240, height: 26))
            let labels = ["Neutral", "Casual messaging", "Formal email", "Clean notes", "Technical / code"]
            popup.addItems(withTitles: labels)
            alert.accessoryView = popup
            NSApp.activate(ignoringOtherApps: true)
            if alert.runModal() == .alertFirstButtonReturn {
                let ids = ["neutral", "casual", "formal", "notes", "code"]
                let picked = ids[max(0, popup.indexOfSelectedItem)]
                var overrides = cfg.dictionary("app_tone_overrides")
                overrides[bundle] = picked
                cfg.set("app_tone_overrides", overrides)
                notify("App tone saved", "\(appName): \(labels[max(0, popup.indexOfSelectedItem)])")
            }
        }
    }

    @objc func cbOpenHub(_ sender: NSMenuItem) {
        guard let store else { return }
        let s = store.stats()
        let rows = store.recent(limit: 25).map {
            "<tr><td>\($0.0)</td><td>\($0.1.replacingOccurrences(of: "<", with: "&lt;"))</td></tr>"
        }.joined()
        let apps = s.topApps.map { "<li>\($0.0): \($0.1)</li>" }.joined()
        let html = """
        <html><head><meta charset="utf-8"><title>Flow Hub</title>
        <style>body{font-family:-apple-system;margin:36px;color:#222}h1{font-size:22px}
        table{border-collapse:collapse;width:100%}td{border-top:1px solid #eee;padding:6px;font-size:13px}
        .k{display:inline-block;margin-right:28px}</style></head><body>
        <h1>Flow Hub</h1>
        <p><span class="k"><b>\(s.count)</b> dictations</span>
        <span class="k"><b>\(s.sumChars / 5)</b> approx words</span>
        <span class="k"><b>\(String(format: "%.1f", s.avgTotal))s</b> avg total</span>
        </p>
        <h2>Top apps</h2><ul>\(apps)</ul>
        <h2>Recent</h2><table>\(rows)</table>
        </body></html>
        """
        let url = FlowConfig.flowDir.appendingPathComponent("hub.html")
        try? html.write(to: url, atomically: true, encoding: .utf8)
        NSWorkspace.shared.open(url)
    }

    @objc func cbOpenKeyboardSettings(_ sender: NSMenuItem) {
        notify("Fn key", "Keyboard settings → \"Press 🌐 key to\" → Do Nothing")
        NSWorkspace.shared.open(
            URL(string: "x-apple.systempreferences:com.apple.Keyboard-Settings.extension")!)
    }

    @objc func cbFreeMemory(_ sender: NSMenuItem) {
        cleanupModel.unload()
        if !recorder.isRecording {
            Task.detached { [self] in await self.transcriber.unload() }
        }
        notify("Memory freed", "Models unloaded, reload on next dictation.")
    }

    @objc func cbEmergencyReset(_ sender: NSMenuItem) {
        if recorder.isRecording { _ = recorder.stop() }
        deferredStopTimer?.cancel()
        deferredStopTimer = nil
        firstFragmentPasted = false
        setState(modelReady ? "idle" : "loading")
        notify("Reset complete", "Ready.")
    }

    @objc func cbStatusClick(_ sender: NSMenuItem) {
        let title = sender.title.lowercased()
        if title.contains("microphone") {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!)
        } else if title.contains("⚠") || title.contains("grant") {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")!)
        }
    }

    @objc func cbKofi(_ sender: NSMenuItem) {
        NSWorkspace.shared.open(URL(string: KOFI_URL)!)
    }

    // MARK: - Launch agent (Launch at Login + keepalive)

    static let launchAgentPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/LaunchAgents/com.shaun.flow.manual.plist")

    func launchAgentInstalled() -> Bool {
        FileManager.default.fileExists(atPath: Self.launchAgentPath.path)
    }

    func appBundlePath() -> String {
        // .../FlowSwift.app/Contents/MacOS/FlowApp -> .../FlowSwift.app
        let exe = Bundle.main.executablePath ?? CommandLine.arguments[0]
        if let r = exe.range(of: ".app/Contents/") {
            return String(exe[..<r.lowerBound]) + ".app"
        }
        return exe
    }

    func setLaunchAtLogin(_ enable: Bool) {
        let plistPath = Self.launchAgentPath
        if enable {
            let bundle = appBundlePath()
            let program = bundle.hasSuffix(".app")
                ? "\(bundle)/Contents/MacOS/\((try? FileManager.default.contentsOfDirectory(atPath: bundle + "/Contents/MacOS").first) ?? "FlowApp")"
                : bundle
            let plist: [String: Any] = [
                "Label": "com.shaun.flow.manual",
                "ProgramArguments": [program],
                "RunAtLoad": true,
                "KeepAlive": ["SuccessfulExit": false],
                "ProcessType": "Interactive",
                "ThrottleInterval": 3,
            ]
            if let data = try? PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0) {
                try? data.write(to: plistPath)
                let p = Process()
                p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
                p.arguments = ["bootstrap", "gui/\(getuid())", plistPath.path]
                try? p.run()
            }
        } else {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            p.arguments = ["bootout", "gui/\(getuid())/com.shaun.flow.manual"]
            try? p.run()
            p.waitUntilExit()
            try? FileManager.default.removeItem(at: plistPath)
        }
    }

    @objc func cbRestart(_ sender: NSMenuItem) {
        let pid = getpid()
        let bundle = appBundlePath()
        let cmd = "while /bin/kill -0 \(pid) 2>/dev/null; do /bin/sleep 0.2; done; /bin/sleep 0.3; /usr/bin/open -a \"\(bundle)\""
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = ["-c", cmd]
        try? p.run()
        NSApp.terminate(nil)
    }

    @objc func cbQuit(_ sender: NSMenuItem) {
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        deferredStopTimer?.cancel()
        deferredStopTimer = nil
        idleUnloadWork?.cancel()
        idleUnloadWork = nil
        if recorder.isRecording { _ = recorder.stop() }
        ipcRecordingContext = nil
        ipcServer?.stop()
        ipcServer = nil
    }
}
