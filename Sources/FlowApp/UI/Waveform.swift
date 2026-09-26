import SwiftUI

/// Forma d'onda a barre alimentata dal livello reale del microfono.
/// Ogni barra e' un campione che scorre verso sinistra, cosi' la voce si "vede"
/// scorrere invece di pulsare a caso.
struct Waveform: View {
    var level: Double            // 0...1
    var active: Bool
    var bars: Int = 22
    var tint: Color = FlowTheme.live
    var height: CGFloat = 22

    @State private var samples: [Double] = []
    @State private var phase: Double = 0

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !active)) { context in
            Canvas { ctx, size in
                let count = max(4, bars)
                let slot = size.width / CGFloat(count)
                let barWidth = max(1.6, slot * 0.52)
                for i in 0..<count {
                    let value = sample(at: i, count: count)
                    let h = max(barWidth, CGFloat(value) * size.height)
                    let x = CGFloat(i) * slot + (slot - barWidth) / 2
                    let rect = CGRect(x: x, y: (size.height - h) / 2, width: barWidth, height: h)
                    let fade = 0.35 + 0.65 * Double(i) / Double(count)   // piu' vivo verso destra
                    ctx.fill(Path(roundedRect: rect, cornerRadius: barWidth / 2),
                             with: .color(tint.opacity(active ? fade : 0.25)))
                }
            }
            .onChange(of: context.date) { _ in push() }
        }
        .frame(height: height)
        .onAppear { if samples.isEmpty { samples = Array(repeating: 0.06, count: bars) } }
    }

    private func sample(at index: Int, count: Int) -> Double {
        guard samples.count == count else { return 0.08 }
        return samples[index]
    }

    /// Scorre la finestra e aggiunge il campione corrente, con un filo di
    /// oscillazione perche' una barra perfettamente piatta sembra un blocco morto.
    private func push() {
        var next = samples
        if next.count != bars { next = Array(repeating: 0.06, count: bars) }
        next.removeFirst()
        // Curva generosa: il parlato normale sta in basso nella scala RMS e
        // barre da 2 px non si leggono in una pillola da 18 punti.
        let shaped = pow(max(0, min(1, level)), 0.62)
        let wobble = active ? (sin(phase) * 0.06 + 0.10) : 0.02
        next.append(min(1, max(0.06, shaped * 0.9 + wobble)))
        samples = next
        phase += 0.55
    }
}

/// Onda a riposo: quattro barre ferme con altezze studiate. Serve un glifo che
/// dica "dettatura" anche da spento, senza sembrare un microfono di sistema.
struct IdleWave: View {
    var tint: Color = .primary
    var height: CGFloat = 13
    var barWidth: CGFloat = 2.1
    var spacing: CGFloat = 2.1
    var heights: [CGFloat] = [0.42, 0.78, 1.0, 0.58]

    var body: some View {
        HStack(alignment: .center, spacing: spacing) {
            ForEach(Array(heights.enumerated()), id: \.offset) { _, h in
                Capsule()
                    .fill(tint)
                    .frame(width: barWidth, height: max(barWidth, height * h))
            }
        }
        .frame(height: height)
    }
}

/// Anello pulsante attorno all'icona quando Flow sta lavorando.
struct BusyHalo: View {
    var tint: Color
    @State private var animate = false

    var body: some View {
        Circle()
            .strokeBorder(tint.opacity(0.85), lineWidth: 1.4)
            .scaleEffect(animate ? 1.35 : 0.85)
            .opacity(animate ? 0 : 0.9)
            .animation(.easeOut(duration: 1.2).repeatForever(autoreverses: false), value: animate)
            .onAppear { animate = true }
    }
}

