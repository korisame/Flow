# Flow

Free, local, unlimited voice dictation for macOS — Whisper turbo on MLX +
optional LLM cleanup. Hold `Fn` to dictate, release to paste.

- Push-to-talk + hands-free auto-segmentation
- Languages: English / Italiano / Français / Русский / العربية
- AI cleanup with local Llama / Qwen 3B–8B or OpenAI Codex CLI
- Cursor-anchored HUD overlay
- ~1.5 GB Whisper turbo, ~1.8 GB optional Qwen 3B (downloaded on first use)

## Install

Download the latest [Flow.dmg](../../releases/latest/download/Flow.dmg),
open it, double-click **Install Flow.command**.

## Build from source

```bash
bash build_dmg.sh
```

Requires Apple Silicon Mac (M1+ or A18+) and Xcode Command Line Tools.

## License

MIT
