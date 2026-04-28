<p align="center">
  <img src="assets/icon-256.png" width="180" alt="Flow icon">
</p>

<h1 align="center">Flow</h1>

<p align="center">
  Free, local, unlimited voice dictation for macOS.<br>
  Whisper turbo on Apple Silicon · optional AI cleanup · hold <kbd>Fn</kbd> to dictate.
</p>

<p align="center">
  <a href="https://github.com/korisame/Flow/releases/latest/download/Flow.dmg">
    <img src="https://img.shields.io/badge/Download-Flow.dmg-white?style=for-the-badge" alt="Download">
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/macOS-12+-black?style=for-the-badge" alt="macOS 12+">
  &nbsp;
  <img src="https://img.shields.io/badge/Apple_Silicon-required-blue?style=for-the-badge" alt="Apple Silicon">
</p>

---

## Features

- **Push-to-talk** — hold `Fn`, release to transcribe and paste
- **Hands-free** — double-press `Fn`, auto-segments by silence
- **Whisper Large-v3 Turbo** on Apple Silicon GPU (MLX)
- **AI cleanup** with local Qwen / Llama (3B, 7B, 8B) or OpenAI Codex CLI
- **5 languages** — English, Italiano, Français, Русский, العربية
- **Cursor-anchored HUD** for recording / transcribing / cleanup state
- **100% offline install** — bundled Python + dependencies in the DMG (~360 MB)

## Install

1. Download the latest [`Flow.dmg`](https://github.com/korisame/Flow/releases/latest/download/Flow.dmg)
2. Open it
3. Right-click **Install Flow.command** → **Open** → confirm
4. Done — Flow is in `/Applications`

The first dictation downloads the Whisper model (~1.5 GB); the optional LLM
for cleanup (~1.8 GB) downloads when you enable it.

## Build from source

```bash
bash build_dmg.sh
```

Requires Apple Silicon Mac, Xcode Command Line Tools (for `clang`), and a
local `~/.flow/venv` (used to pre-bundle Python deps into the DMG).

## License

MIT — see `LICENSE`.
