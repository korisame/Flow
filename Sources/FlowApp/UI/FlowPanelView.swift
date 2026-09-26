import SwiftUI

/// Pannello principale di Flow, aperto dalla voce nella barra dei menu.
struct FlowPanelView: View {
    @Bindable var model: FlowUIModel

    @State private var showSettings = false
    @State private var showHistory = false

    private var look: (color: Color, symbol: String, label: String) { FlowTheme.look(for: model.state) }

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            header
            if model.needsPermission { permissionBanner }
            stage
            tiles
            historySection
            settingsSection
            footer
        }
        .padding(16)
        .frame(width: 380)
        .background(FlowBackdrop().ignoresSafeArea())
        .animation(.smooth(duration: 0.3), value: model.state)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            FlowMark(tint: look.color, active: model.state.isLive)
                .frame(width: 28, height: 28)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text("Flow").font(.system(size: 14, weight: .semibold))
                    Text(APP_VERSION.replacingOccurrences(of: "-swift", with: ""))
                        .font(.system(size: 9.5, weight: .semibold, design: .rounded))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(FlowTheme.accent.opacity(0.18)))
                        .foregroundStyle(FlowTheme.accent)
                }
                HStack(spacing: 5) {
                    Circle().fill(look.color).frame(width: 5, height: 5)
                    Text(model.statusLine)
                        .font(.system(size: 10.5))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }

            Spacer(minLength: 4)

            Button { model.openHub() } label: {
                Image(systemName: "chart.bar.doc.horizontal")
                    .font(.system(size: 12, weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Apri Flow Hub")
        }
    }

    private var permissionBanner: some View {
        Button { model.openPermissions() } label: {
            HStack(spacing: 8) {
                Image(systemName: "lock.trianglebadge.exclamationmark.fill")
                Text("Concedi Accessibilita' e Input Monitoring")
                    .font(.system(size: 11, weight: .medium))
                Spacer()
                Image(systemName: "arrow.up.forward.app").font(.system(size: 10))
            }
            .foregroundStyle(FlowTheme.warn)
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(FlowTheme.warn.opacity(0.14)))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Palco: stato corrente

    private var stage: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                ZStack {
                    Circle().fill(look.color.opacity(0.16))
                    Image(systemName: look.symbol)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(look.color)
                    if model.state.isBusy { BusyHalo(tint: look.color) }
                }
                .frame(width: 34, height: 34)

                VStack(alignment: .leading, spacing: 2) {
                    Text(look.label).font(.system(size: 13, weight: .semibold))
                    if model.state.isLive, let since = model.recordingSince {
                        TimelineView(.periodic(from: .now, by: 0.2)) { ctx in
                            Text(FlowTheme.compactDuration(ctx.date.timeIntervalSince(since)))
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text(hint).font(.system(size: 10.5)).foregroundStyle(.tertiary)
                    }
                }
                Spacer(minLength: 0)

                Waveform(level: model.micLevel, active: model.state.isLive,
                         bars: 16, tint: look.color, height: 26)
                    .frame(width: 92)
                    .opacity(model.state.isLive ? 1 : 0.35)
            }

            if !model.preview.isEmpty, model.state.isLive {
                Text(model.preview)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(.opacity)
            }
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.quaternary.opacity(0.32))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(look.color.opacity(model.state.isLive ? 0.35 : 0.06), lineWidth: 1))
        }
    }

    private var hint: String {
        switch model.state {
        case .loading: "Il modello si sta caricando"
        default: "Fn premuto = detta · doppio Fn = mani libere"
        }
    }

    // MARK: - Numeri

    private var tiles: some View {
        HStack(spacing: 10) {
            InfoTile(icon: "text.quote", label: "dettature",
                     value: "\(model.stats.count)", tint: FlowTheme.accent)
            InfoTile(icon: "textformat.abc", label: "parole",
                     value: model.stats.words > 999
                        ? String(format: "%.1fk", Double(model.stats.words) / 1000)
                        : "\(model.stats.words)",
                     tint: FlowTheme.think)
            InfoTile(icon: "timer", label: "media",
                     value: String(format: "%.1fs", model.stats.avg), tint: FlowTheme.ready)
        }
    }

    // MARK: - Cronologia

    private var historySection: some View {
        VStack(alignment: .leading, spacing: 7) {
            SectionHeader(title: "Ultime dettature",
                          expanded: showHistory,
                          trailing: model.history.isEmpty ? nil : "\(model.history.count)") {
                withAnimation(.snappy(duration: 0.26)) { showHistory.toggle() }
            }

            if let last = model.history.first {
                Button { model.paste(last.text) } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(FlowTheme.accent)
                        Text(last.text)
                            .font(.system(size: 11))
                            .lineLimit(1)
                            .foregroundStyle(.primary)
                        Spacer(minLength: 0)
                        Text("incolla")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 7)
                    .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(.quaternary.opacity(0.3)))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            } else {
                Text("Nessuna dettatura ancora")
                    .font(.system(size: 10.5)).foregroundStyle(.tertiary)
            }

            if showHistory {
                VStack(spacing: 3) {
                    ForEach(Array(model.history.dropFirst().enumerated()), id: \.offset) { _, row in
                        Button { model.paste(row.text) } label: {
                            HStack(spacing: 6) {
                                Text(row.text)
                                    .font(.system(size: 10.5))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Spacer(minLength: 0)
                            }
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(HoverRowStyle())
                    }
                    if model.history.count > 1 {
                        HStack {
                            Spacer()
                            Button("Svuota cronologia") { model.clearHistory() }
                                .buttonStyle(.plain)
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)
                        }
                        .padding(.top, 2)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    // MARK: - Impostazioni

    private var settingsSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            SectionHeader(title: "Impostazioni", expanded: showSettings, trailing: nil) {
                withAnimation(.snappy(duration: 0.26)) { showSettings.toggle() }
            }

            if showSettings {
                VStack(alignment: .leading, spacing: 11) {
                    labeled("Rifinitura AI") {
                        ForEach(AI_BACKENDS, id: \.0) { id, label in
                            Chip(label: label.components(separatedBy: "  ·  ").first ?? label,
                                 selected: model.aiBackend == id) { model.setBackend(id) }
                        }
                    }
                    Text("FullStop corregge la punteggiatura senza riscrivere le parole.")
                        .font(.caption).foregroundStyle(.secondary)

                    VStack(spacing: 6) {
                        ForEach(FlowPanelView.switches, id: \.key) { row in
                            SwitchRow(title: row.title, subtitle: row.subtitle,
                                      isOn: model.flag(row.key)) { model.toggle(row.key) }
                        }
                    }

                    HStack(spacing: 6) {
                        Button { model.editDictionary() } label: {
                            Label("Dizionario", systemImage: "character.book.closed")
                                .font(.system(size: 10.5, weight: .medium))
                        }
                        .buttonStyle(SoftButtonStyle(tint: FlowTheme.think))
                        Button { model.suggestDictionary() } label: {
                            Label("Suggerisci", systemImage: "wand.and.stars")
                                .font(.system(size: 10.5, weight: .medium))
                        }
                        .buttonStyle(SoftButtonStyle(tint: FlowTheme.think))
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    static let switches: [(key: String, title: String, subtitle: String?)] = [
        ("instant_paste", "Incolla immediato", "poi rifinisce il testo sul posto"),
        ("merge_bursts", "Unisci dettature vicine", "punteggiatura continua entro 25s"),
        ("live_preview", "Anteprima dal vivo", nil),
        ("remove_fillers", "Togli intercalari", nil),
        ("verbal_commands", "Comandi vocali", nil),
        ("use_context", "Usa il contesto dell'app", nil),
        ("sound_feedback", "Suoni", nil),
        ("show_hud", "Mostra HUD", nil),
        ("verify_paste", "Verifica l'incolla", nil),
        ("keep_cleanup_warm", "Modello di rifinitura caldo", "piu' veloce, piu' RAM"),
        ("free_model_idle", "Libera il modello da fermo", "meno RAM"),
        ("launch_at_login", "Avvia al login", nil),
    ]

    private func labeled<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title.uppercased())
                .font(.system(size: 8.5, weight: .semibold))
                .foregroundStyle(.tertiary)
            FlowWrap(spacing: 5) { content() }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 7) {
            Button { model.pasteLast() } label: {
                Label("Incolla ultimo", systemImage: "arrow.down.doc")
                    .font(.system(size: 11, weight: .medium))
            }
            .buttonStyle(SoftButtonStyle(tint: FlowTheme.accent, prominent: true))
            .disabled(model.history.isEmpty)

            Button { model.freeMemory() } label: {
                Image(systemName: "memorychip").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(SoftButtonStyle(tint: .secondary))
            .help("Libera memoria (scarica i modelli)")

            Button { model.reset() } label: {
                Image(systemName: "stop.circle").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(SoftButtonStyle(tint: .secondary))
            .help("Stop / reset")

            Spacer(minLength: 0)

            Button { model.kofi() } label: {
                Image(systemName: "cup.and.saucer").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(SoftButtonStyle(tint: .secondary))
            .help("Offri un caffe' su Ko-fi")

            Button { model.restart() } label: {
                Image(systemName: "arrow.clockwise").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(SoftButtonStyle(tint: .secondary))
            .help("Riavvia Flow")

            Button { model.quit() } label: {
                Image(systemName: "power").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(SoftButtonStyle(tint: .secondary))
            .help("Esci")
        }
    }
}

// MARK: - Pezzi di supporto

struct SectionHeader: View {
    let title: String
    let expanded: Bool
    let trailing: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Text(title.uppercased())
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.tertiary)
                if let trailing {
                    Text(trailing)
                        .font(.system(size: 8.5, weight: .semibold, design: .rounded))
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(Capsule().fill(Color.primary.opacity(0.08)))
                        .foregroundStyle(.tertiary)
                }
                Image(systemName: "chevron.down")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.quaternary)
                    .rotationEffect(.degrees(expanded ? 180 : 0))
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct HoverRowStyle: ButtonStyle {
    @State private var hovering = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(hovering ? Color.primary.opacity(0.07) : .clear))
            .onHover { hovering = $0 }
            .opacity(configuration.isPressed ? 0.6 : 1)
    }
}

/// Glifo di Flow: onda dentro un cerchio, che respira quando registra.
struct FlowMark: View {
    var tint: Color
    var active: Bool
    @State private var pulse = false

    var body: some View {
        ZStack {
            Circle()
                .fill(LinearGradient(colors: [tint.opacity(0.30), tint.opacity(0.12)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
            Image(systemName: "waveform")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(tint)
                .scaleEffect(active && pulse ? 1.12 : 1)
        }
        .animation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true), value: pulse)
        .onAppear { pulse = active }
        .onChange(of: active) { _, newValue in pulse = newValue }
    }
}
