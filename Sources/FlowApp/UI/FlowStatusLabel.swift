import SwiftUI

/// Contenuto della voce di Flow nella barra dei menu: glifo di stato,
/// forma d'onda dal vivo mentre registra e cronometro.
struct FlowStatusLabel: View {
    @Bindable var model: FlowUIModel

    private var look: (color: Color, symbol: String, label: String) { FlowTheme.look(for: model.state) }

    var body: some View {
        HStack(spacing: 4) {
            ZStack {
                switch model.state {
                case .idle:
                    // A riposo la firma di Flow e' l'onda ferma: si distingue
                    // dagli anelli delle altre barre e resta leggibile ovunque.
                    IdleWave(tint: Color.primary.opacity(0.85), height: 13)
                case .loading:
                    IdleWave(tint: Color.primary.opacity(0.35), height: 13)
                default:
                    Image(systemName: look.symbol)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(look.color)
                }
                if model.state.isBusy {
                    BusyHalo(tint: look.color).frame(width: 18, height: 18)
                }
            }
            .frame(width: 17, height: 16)

            if model.state.isLive {
                Waveform(level: model.micLevel, active: true, bars: 9,
                         tint: look.color, height: 13)
                    .frame(width: 26)

                if let since = model.recordingSince {
                    TimelineView(.periodic(from: .now, by: 0.5)) { ctx in
                        Text("\(Int(ctx.date.timeIntervalSince(since)))s")
                            .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(look.color)
                    }
                }
            }
        }
        .padding(.horizontal, 3)
        .frame(height: 22)
        .fixedSize()
        .animation(.snappy(duration: 0.3), value: model.state)
    }
}

