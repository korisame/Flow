import Foundation
import AppKit
import ApplicationServices

public struct AppContext {
    public var bundleId: String?
    public var appName: String?
    public var url: String?
    public var focusedRole: String?
    public var focusedText: String?
    public init() {}
}

public struct PasteResult {
    public var ok: Bool?      // nil = unverified
    public var method: String
    public var error: String?
    public var pasteS: Double = 0
    /// PID that owned the focused element when the text was pasted. Used to
    /// prove we are still looking at the same target before correcting it.
    public var targetPid: pid_t = 0

    public init(ok: Bool? = nil, method: String, error: String? = nil,
                pasteS: Double = 0, targetPid: pid_t = 0) {
        self.ok = ok
        self.method = method
        self.error = error
        self.pasteS = pasteS
        self.targetPid = targetPid
    }
}

/// Paster: pasteboard + synthetic Cmd+V, with clipboard save/restore and
/// AX verification. Port of flow.py _paste / _verify_paste_result.
public final class Paster {
    private var clipboardSaved: String? = nil
    private var restoreTimer: Timer? = nil
    private let lock = NSLock()
    public var verifyPaste = true

    public init() {}

    // MARK: - Context capture

    public func captureContext(includeText: Bool) -> AppContext {
        var ctx = AppContext()
        if let app = NSWorkspace.shared.frontmostApplication {
            ctx.bundleId = app.bundleIdentifier
            ctx.appName = app.localizedName
        }
        if includeText {
            let (role, text) = focusedElementInfo()
            ctx.focusedRole = role
            ctx.focusedText = text
        }
        if let b = ctx.bundleId, ["com.apple.Safari", "com.google.Chrome"].contains(b) {
            ctx.url = browserURL(bundleId: b)
        }
        return ctx
    }

    private func browserURL(bundleId: String) -> String? {
        let script = bundleId == "com.apple.Safari"
            ? "tell application \"Safari\" to get URL of current tab of front window"
            : "tell application \"Google Chrome\" to get URL of active tab of front window"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        p.arguments = ["-e", script]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do { try p.run() } catch { return nil }
        let deadline = Date().addingTimeInterval(0.8)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        if p.isRunning { p.terminate(); return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let s = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (s?.isEmpty ?? true) ? nil : s
    }

    private func focusedElement() -> AXUIElement? {
        let system = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let el = focused else { return nil }
        return (el as! AXUIElement)
    }

    /// PID owning the currently focused element. Unlike
    /// NSWorkspace.frontmostApplication this is a direct AX query, so it is
    /// accurate when called from the worker thread.
    private func focusedPid() -> pid_t {
        guard let element = focusedElement() else { return 0 }
        var pid: pid_t = 0
        return AXUIElementGetPid(element, &pid) == .success ? pid : 0
    }

    private func focusedElementInfo() -> (String?, String?) {
        guard let element = focusedElement() else { return (nil, nil) }
        var roleRef: CFTypeRef?
        var role: String? = nil
        if AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleRef) == .success {
            role = roleRef as? String
        }
        var valueRef: CFTypeRef?
        var text: String? = nil
        if AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success {
            if let s = valueRef as? String {
                text = String(s.suffix(2000))
            }
        }
        return (role, text)
    }

    // MARK: - Clipboard

    private func readClipboard() -> String? {
        NSPasteboard.general.string(forType: .string)
    }

    private func writeClipboard(_ s: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(s, forType: .string)
    }

    private func scheduleRestore() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.restoreTimer?.invalidate()
            self.restoreTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) { _ in
                self.lock.lock()
                let saved = self.clipboardSaved
                self.clipboardSaved = nil
                self.lock.unlock()
                if let saved { self.writeClipboard(saved) }
            }
        }
    }

    /// PID that owns the element with keyboard focus right now.
    public func focusedTargetPid() -> pid_t { focusedPid() }

    private func selectedRange(of element: AXUIElement) -> CFRange? {
        var rangeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString,
                                            &rangeRef) == .success,
              let axValue = rangeRef, CFGetTypeID(axValue) == AXValueGetTypeID() else { return nil }
        var range = CFRange()
        guard AXValueGetValue(axValue as! AXValue, .cfRange, &range) else { return nil }
        return range
    }

    /// Put the insertion point back at the end of the text after a whole-value
    /// rewrite, so the next paste appends instead of landing at the top.
    /// Only done when the caret was at the end to begin with: if the user had
    /// deliberately clicked somewhere else, leave their cursor alone.
    private func restoreCaret(in element: AXUIElement, wasAtEnd: Bool, newLength: Int) {
        guard wasAtEnd else { return }
        var range = CFRange(location: newLength, length: 0)
        guard let axValue = AXValueCreate(.cfRange, &range) else { return }
        let status = AXUIElementSetAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, axValue)
        if status != .success {
            flowLog("[paste] could not restore caret after rewrite (AX \(status.rawValue))")
        }
    }

    /// Where the insertion point is, for diagnosing out-of-order pastes.
    public func describeCaret() -> String {
        guard let element = focusedElement() else { return "caret: no focused element" }
        var valueRef: CFTypeRef?
        let value = AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success
            ? (valueRef as? String) : nil
        var roleRef: CFTypeRef?
        let role = AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleRef) == .success
            ? (roleRef as? String) : nil
        var rangeRef: CFTypeRef?
        var caret = "n/a"
        if AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeRef) == .success,
           let axValue = rangeRef, CFGetTypeID(axValue) == AXValueGetTypeID() {
            var range = CFRange()
            if AXValueGetValue(axValue as! AXValue, .cfRange, &range) {
                caret = "loc=\(range.location) len=\(range.length)"
            }
        }
        return "caret: role=\(role ?? "?") textLen=\(value?.count ?? -1) \(caret)"
    }

    /// Does the focused field still end with exactly this text? Used to decide
    /// whether a previous dictation is still intact and can be extended.
    public func focusedTextEndsWith(_ text: String) -> Bool {
        guard !text.isEmpty, let element = focusedElement() else { return false }
        var valueRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success,
              let current = valueRef as? String else { return false }
        return current.hasSuffix(text)
    }

    /// Put `text` on the clipboard and keep it there, so the user can recover a
    /// dictation that could not be placed with a plain Cmd+V.
    public func stageOnClipboard(_ text: String) {
        writeClipboard(text)
        keepTextOnClipboard()
    }

    /// Leave the transcript on the clipboard instead of restoring the previous
    /// contents, and cancel any restore already armed by an earlier fragment of
    /// the same burst. Used when the paste could not be confirmed: the text is
    /// then one Cmd+V away instead of being silently discarded.
    private func keepTextOnClipboard() {
        lock.lock()
        clipboardSaved = nil
        lock.unlock()
        DispatchQueue.main.async { [weak self] in
            self?.restoreTimer?.invalidate()
            self?.restoreTimer = nil
        }
    }

    // MARK: - Key synthesis

    private func postKey(_ keyCode: CGKeyCode, flags: CGEventFlags) {
        guard let src = CGEventSource(stateID: .hidSystemState) else { return }
        if let down = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true) {
            down.flags = flags
            down.post(tap: .cghidEventTap)
        }
        if let up = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false) {
            up.flags = flags
            up.post(tap: .cghidEventTap)
        }
    }

    public func undo() {
        postKey(6, flags: .maskCommand)  // Cmd+Z
    }

    public func sendBackspaces(_ n: Int) {
        guard n > 0 else { return }
        for _ in 0..<n {
            postKey(51, flags: [])  // kVK_Delete
            usleep(1_000)
        }
    }

    // MARK: - Paste

    /// Roles that cannot receive text. Cmd+V into one of these does nothing at
    /// all, so send the transcript to the clipboard rather than into a menu.
    /// Menu roles only: unambiguous, and both observed cases were AXMenuItem.
    /// Deliberately excludes AXButton — some web/Electron composers report it
    /// while a real editable field has focus.
    private static let nonEditableRoles: Set<String> = [
        "AXMenuItem", "AXMenu", "AXMenuBar", "AXMenuBarItem",
    ]

    public func paste(_ text: String, contextBefore: AppContext? = nil) -> PasteResult {
        let t0 = Date()
        var result = PasteResult(ok: nil, method: "unverified", error: nil)
        // Re-read the focused element NOW. `contextBefore` was captured before
        // the cleanup LLM ran and is seconds stale, so using its text as the
        // length baseline compared against the wrong field contents.
        let (roleNow, textNow) = verifyPaste ? focusedElementInfo() : (nil, nil)
        var before = contextBefore ?? captureContext(includeText: verifyPaste)
        if verifyPaste { before.focusedText = textNow }

        // Save clipboard once per burst
        lock.lock()
        if clipboardSaved == nil {
            clipboardSaved = readClipboard() ?? ""
        }
        lock.unlock()

        writeClipboard(text)

        if let role = roleNow, Self.nonEditableRoles.contains(role) {
            keepTextOnClipboard()
            var r = PasteResult(ok: false, method: "target-not-editable",
                                error: "focused element is \(role) — text left on the clipboard")
            r.pasteS = -t0.timeIntervalSinceNow
            return r
        }

        let pid = focusedPid()
        Thread.sleep(forTimeInterval: 0.06)
        postKey(9, flags: .maskCommand)  // Cmd+V
        Thread.sleep(forTimeInterval: 0.12)

        result = verifyResult(text: text, before: before)
        result.targetPid = pid
        if result.ok == false {
            // Genuinely not confirmed: keep the transcript reachable with Cmd+V
            // instead of restoring the old clipboard over it.
            keepTextOnClipboard()
        } else {
            scheduleRestore()
        }
        result.pasteS = -t0.timeIntervalSinceNow
        return result
    }

    // MARK: - Two-phase paste (draft now, polish later)

    public enum ReplaceOutcome: String {
        case replaced          // corrected in place
        case identical         // nothing to change
        case focusChanged      // user moved on: draft left alone
        case textChanged       // user kept typing: draft left alone
        case caretMoved        // insertion point left the text we pasted
        case tooLongToRetype   // no settable AX value and too big to backspace
        case unavailable       // no AX access
    }

    /// Replace a draft that was just pasted with its cleaned-up version.
    ///
    /// This only ever runs when we can prove the draft is still the tail of the
    /// focused field, so a user who kept typing, switched app, or sent the
    /// message never gets characters eaten. When it cannot prove that, the
    /// draft simply stays — it is valid text, just missing a few commas.
    public func replaceDraft(_ draft: String, with final: String,
                             expectedPid: pid_t) -> ReplaceOutcome {
        if draft == final { return .identical }
        guard let element = focusedElement() else { return .unavailable }
        if expectedPid != 0 {
            var pid: pid_t = 0
            guard AXUIElementGetPid(element, &pid) == .success, pid == expectedPid else {
                return .focusChanged
            }
        }

        var valueRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success,
              let current = valueRef as? String else { return .unavailable }

        // The draft must still be exactly what sits at the end of the field.
        guard current.hasSuffix(draft) else { return .textChanged }

        // Preferred path: rewrite the value directly. Instant, atomic, and it
        // cannot race the keyboard. Plain AppKit text fields support it.
        var settable = DarwinBoolean(false)
        if AXUIElementIsAttributeSettable(element, kAXValueAttribute as CFString, &settable) == .success,
           settable.boolValue {
            let caretBefore = selectedRange(of: element)
            let rewritten = String(current.dropLast(draft.count)) + final
            if AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString,
                                            rewritten as CFTypeRef) == .success {
                // Replacing the whole value resets the insertion point in most
                // implementations, usually to 0. Leaving it there sent the NEXT
                // dictation to the top of the field, above the previous one.
                restoreCaret(in: element, wasAtEnd: caretBefore.map { $0.location >= current.count } ?? true,
                             newLength: rewritten.count)
                return .replaced
            }
        }

        // Fallback for Electron/WebKit composers, which expose the value but
        // refuse to have it set: retype the tail. Bounded, so a long dictation
        // never turns into a thousand synthetic keystrokes.
        guard draft.count <= 300 else { return .tooLongToRetype }

        // Backspaces delete whatever sits before the insertion point, so they
        // are only safe when the insertion point is still at the end of the
        // text we pasted. If the app moved it, or the user selected something,
        // deleting here would eat the wrong characters.
        if let caret = selectedRange(of: element) {
            guard caret.length == 0, caret.location == current.count else {
                return .caretMoved
            }
        }

        lock.lock()
        if clipboardSaved == nil { clipboardSaved = readClipboard() ?? "" }
        lock.unlock()

        sendBackspaces(draft.count)
        Thread.sleep(forTimeInterval: 0.05)
        writeClipboard(final)
        Thread.sleep(forTimeInterval: 0.06)
        postKey(9, flags: .maskCommand)  // Cmd+V
        Thread.sleep(forTimeInterval: 0.12)
        scheduleRestore()
        return .replaced
    }

    /// Poll the focused element instead of reading it once. Electron and
    /// WebKit composers (Claude, ChatGPT, Safari) publish the new AX value tens
    /// to hundreds of ms after Cmd+V, so a single early read reported perfectly
    /// good pastes as failures. Returns as soon as the text is visible, so the
    /// common case stays as fast as before.
    private func verifyResult(text: String, before: AppContext) -> PasteResult {
        if !verifyPaste { return PasteResult(ok: nil, method: "unverified", error: nil) }
        if text.isEmpty { return PasteResult(ok: true, method: "empty", error: nil) }

        func normalize(_ s: String) -> String {
            s.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).joined(separator: " ").lowercased()
        }
        let needle = normalize(String(text.suffix(180)))
        let beforeLen = before.focusedText?.count ?? 0

        let deadline = Date().addingTimeInterval(1.2)
        var sawAX = false
        repeat {
            let (_, afterText) = focusedElementInfo()
            if let after = afterText {
                sawAX = true
                let haystack = normalize(after)
                if !needle.isEmpty && haystack.contains(needle) {
                    // Landing anywhere other than the end means the insertion
                    // point was not where the previous dictation left it, which
                    // is how a follow-up ends up above the text it should follow.
                    let atEnd = haystack.hasSuffix(needle)
                    return PasteResult(ok: true,
                                       method: atEnd ? "ax-text-match" : "ax-text-match-midfield",
                                       error: nil)
                }
                if after.count > beforeLen + max(1, min(text.count, 20)) {
                    return PasteResult(ok: true, method: "ax-length-delta", error: nil)
                }
            }
            if Date() >= deadline { break }
            Thread.sleep(forTimeInterval: 0.08)
        } while true

        guard sawAX else {
            return PasteResult(ok: nil, method: "ax-unavailable", error: nil)
        }
        return PasteResult(ok: false, method: "ax-text-missing",
                           error: "pasted text not found in focused field")
    }
}

