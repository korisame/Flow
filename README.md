# Flow

Local voice dictation for macOS, reconstructed in Swift.

This repository contains the current Swift app, CLI, Core ML/FluidAudio
transcriber, local cleanup pipeline, IPC layer, history store, and tests. The
older Python launcher previously published in this repository was replaced by
this Swift source tree.

## Status

This checkout is a reconstructed source tree, not a byte-for-byte recovery of
the installed application. `RECOVERY.md` records the recovery boundary and
the verification performed on 21 September 2026. The active transcription
backend uses the local Parakeet TDT 0.6B v3 path through FluidAudio; cleanup is
local and failures preserve the original text.

## Build and test

Requires macOS, Swift/Xcode tooling, and the pinned FluidAudio dependency.

```sh
swift test
swift build -c release
```

To build a fresh app staging bundle after the release build:

```sh
CODESIGN_IDENTITY="-" ./make-app.sh
```

Use a real local signing identity only on the development machine. No signing
identity, model weights, audio, history database, or runtime build output is
part of this repository.

## License

MIT — see `LICENSE`.
