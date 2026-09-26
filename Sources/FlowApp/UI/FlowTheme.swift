import SwiftUI

/// Linguaggio visivo di Flow, allineato a Claude Usage Bar:
/// materiale di sistema, accenti caldi, tipografia rounded, animazioni molleggiate.
enum FlowTheme {
    static let accent  = Color(red: 0.85, green: 0.47, blue: 0.34)   // coral
    static let live    = Color(red: 0.94, green: 0.30, blue: 0.33)   // rosso registrazione
    static let ready   = Color(red: 0.18, green: 0.80, blue: 0.66)   // teal
    static let think   = Color(red: 0.52, green: 0.60, blue: 0.98)   // indaco elaborazione
    static let warn    = Color(red: 0.96, green: 0.72, blue: 0.31)

    static let track = Color.primary.opacity(0.16)
    static let barTrack = Color.primary.opacity(0.28)

    /// Colore e simbolo per ogni stato della pipeline.
    static func look(for state: FlowState) -> (color: Color, symbol: String, label: String) {
        switch state {
        case .loading:  (warn, "circle.dotted", "Caricamento modello")
        case .idle:     (ready, "mic.fill", "Pronto")
        case .recording:(live, "mic.fill", "In ascolto")
        case .handsFree:(live, "infinity", "Mani libere")
        case .transcribing: (think, "waveform", "Trascrizione")
        case .cleaning: (accent, "sparkles", "Rifinitura")
        }
    }

    static func compactDuration(_ seconds: TimeInterval) -> String {
        let s = max(0, Int(seconds))
        let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(sec)s" }
        return "\(sec)s"
    }
}

/// Stati della pipeline, mappati sulle stringhe usate dall'AppDelegate.
enum FlowState: String, Sendable, Equatable {
    case loading, idle, recording, handsFree, transcribing, cleaning

    init(legacy: String) {
        switch legacy {
        case "rec": self = .recording
        case "rec_hf": self = .handsFree
        case "proc": self = .transcribing
        case "ai": self = .cleaning
        case "load", "loading": self = .loading
        default: self = .idle
        }
    }

    var isLive: Bool { self == .recording || self == .handsFree }
    var isBusy: Bool { self == .transcribing || self == .cleaning || self == .loading }
}

// MARK: - Componenti condivisi

struct SoftButtonStyle: ButtonStyle {
    var tint: Color
    var prominent: Bool = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(tint.opacity(configuration.isPressed ? 0.34 : (prominent ? 0.22 : 0.14)))
            )
            .foregroundStyle(tint)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.snappy(duration: 0.15), value: configuration.isPressed)
    }
}

struct Chip: View {
    let label: String
    let selected: Bool
    var tint: Color = FlowTheme.accent
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 10.5, weight: selected ? .semibold : .regular))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Capsule().fill(selected ? tint.opacity(0.22) : Color.primary.opacity(0.06)))
                .overlay(Capsule().strokeBorder(selected ? tint.opacity(0.5) : .clear, lineWidth: 1))
                .foregroundStyle(selected ? tint : .secondary)
        }
        .buttonStyle(.plain)
        .animation(.snappy(duration: 0.2), value: selected)
    }
}

/// Interruttore compatto: riga cliccabile con pallino animato.
struct SwitchRow: View {
    let title: String
    let subtitle: String?
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 11.5, weight: .medium))
                    if let subtitle {
                        Text(subtitle).font(.system(size: 9.5)).foregroundStyle(.tertiary)
                    }
                }
                Spacer(minLength: 6)
                ZStack(alignment: isOn ? .trailing : .leading) {
                    Capsule()
                        .fill(isOn ? FlowTheme.ready.opacity(0.55) : Color.primary.opacity(0.14))
                        .frame(width: 28, height: 16)
                    Circle()
                        .fill(.white)
                        .frame(width: 12, height: 12)
                        .shadow(radius: 1, y: 0.5)
                        .padding(.horizontal, 2)
                }
                .frame(width: 28, height: 16)
            }
            .contentShape(Rectangle())
            .foregroundStyle(.primary)
        }
        .buttonStyle(.plain)
        .animation(.snappy(duration: 0.22), value: isOn)
    }
}

struct InfoTile: View {
    let icon: String
    let label: String
    let value: String
    let tint: Color

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 0) {
                Text(label.uppercased())
                    .font(.system(size: 8.5, weight: .semibold))
                    .foregroundStyle(.tertiary)
                Text(value)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 11, style: .continuous).fill(.quaternary.opacity(0.3)))
    }
}

/// Sfondo del pannello: materiale di sistema piu' un velo caldo.
struct FlowBackdrop: View {
    var body: some View {
        ZStack {
            Rectangle().fill(.ultraThinMaterial)
            LinearGradient(colors: [FlowTheme.accent.opacity(0.10), .clear, FlowTheme.think.opacity(0.07)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }
}

/// Layout a capo automatico per i chip.
struct FlowWrap: Layout {
    var spacing: CGFloat = 5

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? 320
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0; y += rowHeight + spacing; rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: maxWidth, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX; y += rowHeight + spacing; rowHeight = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

