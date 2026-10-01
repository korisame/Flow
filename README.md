# Flow

Current version: **2.4.0**. Native macOS SwiftUI dictation app, local Core ML transcription and offline punctuation cleanup.

## Build and verify

Requires macOS, Swift/Xcode tooling and the pinned FluidAudio 0.17.3 dependency.

```sh
swift test
swift build -c release
CODESIGN_IDENTITY="-" ./make-app.sh
```

Transcription uses Parakeet Ultra Core ML pinned to revision `95eaa59a39d4394f047a4dc5cce480388a60d1b6`. The cleanup worker keeps processing local and preserves original text on failure. Model weights, recordings, transcript databases, local settings and compiled binaries are excluded.

The source was reconstructed in September 2026 and subsequently updated to Ultra. The 2.4.0 source and installed app version were reconciled against local sessions on 1 October 2026. Historical recovery limitations are documented in RECOVERY.md.
