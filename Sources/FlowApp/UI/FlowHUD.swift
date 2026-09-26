import AppKit
import SwiftUI
import FlowCore

/// Pillola che compare vicino al cursore mentre Flow ascolta o elabora.
struct FlowHUDView: View {
    var state: FlowState
    var level: Double
    var preview: String
    var since: Date?

    private var look: (color: Color, symbol: String, label: String) { FlowTheme.look(for: state) }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 9) {
                ZStack {
                    Circle().fill(look.color.opacity(0.18))
                    Image(systemName: look.symbol)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(look.color)
                    if state.isBusy { BusyHalo(tint: look.color) }
                }
                .frame(width: 24, height: 24)

                Text(look.label)
                    .font(.system(size: 12.5, weight: .semibold))
                    .fixedSize()

                if state.isLive {
                    Waveform(level: level, active: true, bars: 14, tint: look.color, height: 18)
                        .frame(width: 62)
                    if let since {
                        TimelineView(.periodic(from: .now, by: 0.25)) { ctx in
                            Text(FlowTheme.compactDuration(ctx.date.timeIntervalSince(since)))
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer(minLength: 0)
            }

            if !preview.isEmpty {
                Text(preview)
                    .font(.system(size: 11.5))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.head)
                    .multilineTextAlignment(.leading)
                    // Larghezza fissa: senza questo la pillola si allunga
                    // fino a tutto lo schermo appena la frase cresce.
                    .frame(width: 360, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 10)
        .background {
            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous).fill(.ultraThinMaterial)
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(LinearGradient(colors: [look.color.opacity(0.16), .clear],
                                         startPoint: .topLeading, endPoint: .bottomTrailing))
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(look.color.opacity(0.30), lineWidth: 1)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .shadow(color: .black.opacity(0.28), radius: 14, y: 6)
        .animation(.snappy(duration: 0.28), value: state)
    }
}

/// Contenitore AppKit dell'HUD: pannello borderless che segue il cursore.
/// Stessa interfaccia della versione precedente, cosi' il resto dell'app non cambia:
/// ogni metodo pubblico puo' essere chiamato da qualsiasi thread e rimbalza su main.
final class HudPanel {
    private var panel: NSPanel?
    private var hosting: NSHostingView<FlowHUDView>?
    private var preview = ""
    private var state: FlowState = .idle
    private var since: Date?
    private var levelTimer: Timer?
    private var level: Double = 0
    weak var recorder: Recorder?

    func setPreview(_ text: String, state: String) {
        DispatchQueue.main.async { [self] in
            preview = text
            show(state: state)
        }
    }

    func clearPreview() {
        DispatchQueue.main.async { [self] in preview = "" }
    }

    func show(state rawState: String) {
        DispatchQueue.main.async { [self] in
            guard FlowConfig().bool("show_hud", default: true) else { return }
            let newState = FlowState(legacy: rawState)
            if newState.isLive, since == nil { since = .now }
            if !newState.isLive { since = nil }
            state = newState

            let p = ensurePanel()
            render()
            let size = hosting?.fittingSize ?? NSSize(width: 200, height: 48)
            p.setContentSize(size)

            let mouse = NSEvent.mouseLocation
            var x = mouse.x + 18
            var y = mouse.y - size.height - 12
            if let screen = NSScreen.screens.first(where: { $0.frame.contains(mouse) }) ?? NSScreen.main {
                x = min(max(screen.frame.minX + 8, x), screen.frame.maxX - size.width - 8)
                y = min(max(screen.frame.minY + 8, y), screen.frame.maxY - size.height - 8)
            }
            p.setFrameOrigin(NSPoint(x: x, y: y))
            p.orderFrontRegardless()
            startLevelPolling()
        }
    }

    func hide() {
        DispatchQueue.main.async { [self] in
            stopLevelPolling()
            since = nil
            level = 0
            panel?.orderOut(nil)
        }
    }

    // MARK: - Interno

    private func ensurePanel() -> NSPanel {
        if let panel { return panel }
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 200, height: 48),
                        styleMask: [.borderless, .nonactivatingPanel],
                        backing: .buffered, defer: true)
        p.level = .screenSaver
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = false
        p.ignoresMouseEvents = true
        p.collectionBehavior = [.canJoinAllSpaces, .transient, .fullScreenAuxiliary]
        let view = NSHostingView(rootView: currentView())
        p.contentView = view
        hosting = view
        panel = p
        return p
    }

    private func currentView() -> FlowHUDView {
        FlowHUDView(state: state,
                    level: level,
                    preview: state.isLive ? preview : "",
                    since: since)
    }

    private func render() {
        hosting?.rootView = currentView()
    }

    private func startLevelPolling() {
        guard state.isLive, levelTimer == nil else { return }
        levelTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { [weak self] _ in
            guard let self, let rec = self.recorder else { return }
            let newLevel = Double(rec.inputLevel)
            guard abs(newLevel - self.level) > 0.01 else { return }
            self.level = newLevel
            self.render()
        }
    }

    private func stopLevelPolling() {
        levelTimer?.invalidate()
        levelTimer = nil
    }
}

