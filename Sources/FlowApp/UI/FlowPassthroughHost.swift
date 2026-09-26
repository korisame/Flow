import AppKit
import SwiftUI

/// NSHostingView trasparente ai click: il bottone della barra dei menu resta
/// cliccabile e continua a distinguere click sinistro e destro.
final class FlowPassthroughHost<Content: View>: NSHostingView<Content> {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

