# Flow v1.1.0

## Streaming dictation con qualità batch

Le frasi appaiono mentre detti, ma il risultato finale ha la stessa qualità della modalità batch (anzi superiore in molti casi). Il light AI cleanup gira in background ogni 2 secondi di pausa, sostituendo il testo grezzo con la versione pulita. Al rilascio del tasto fn, una passata heavy sull'intero audio garantisce coerenza, capitalizzazione, punteggiatura e fusione dei boundary.

## Cosa cambia rispetto a v1.0.0

### Streaming dictation (nuovo)
- Live preview: il testo appare mentre detti, una frase per volta, su pause naturali rilevate da Silero VAD.
- Light cleanup intermedio: dopo 2 secondi di pausa, l'AI ripulisce in background il testo già pasted, rimuovendo punteggiatura spuria e fillers, fondendo eventuali duplicazioni ai boundary.
- Final batch al rilascio fn: l'audio completo viene ritrascritto da Whisper in un solo decode, poi una cleanup AI heavy normalizza tutto. Niente più `questa... questa` o frasi spezzate da pause naturali.
- Toggle in Settings → "Streaming dictation (paste as you speak)".

### Robustezza
- Fix del bug fn-key release stuck: il callback dell'event tap ora ritorna in microsecondi (handlers async + listen-only mode), niente più stati "registrazione infinita".
- Fix race del clipboard: i paste in rapida successione non causano più paste di clipboard vecchi (debounced restore 1s dopo l'ultimo paste).
- Fix shape mismatch numpy nell'audio buffer (il recorder si fermava silenziosamente dopo il primo VAD flush).
- Stop / Reset menu item per recovery manuale da stati bloccati.

### Qualità AI cleanup
- Prompt esplicito su no-em-dash, niente connettivi inventati ("Ok", "Allora", "So"), no traduzioni, fusione duplicazioni boundary.
- Few-shot italiani aggiornati con esempi di puntini-respiro.
- Skip-when-clean bypassato sul finale streaming (il testo accumulato è SEMPRE da pulire).

### Anti-allucinazione
- Strip stock phrase per allucinazioni Whisper italiane: "Ti sto mostrando come fare il video", "Sottotitoli a cura di...", "[Musica]", "[Applausi]", e altre 10 pattern. Anchored a fine stringa per evitare falsi positivi.
- Multilingual stock phrase strip esistente esteso (inglese, francese, spagnolo, tedesco, russo).

### Lingue
- Lang lock: dopo il primo segmento, la lingua resta fissa per tutta la sessione (niente più "Ok non funziona" classificato come tedesco).
- Hybrid mode auto-promote turbo → large-v3 per ru/ar (opt-in, scaricato in background).

### Performance
- Pre-warm large-v3 in background quando l'utente seleziona ru/ar.
- Pre-warm LLM locale (Qwen, Llama) all'avvio.
- AI lock serializza chiamate concorrenti al modello (mlx_lm non è thread-safe).

### UX
- Settings submenu pulito (engine, AI cleanup, tone, dictionary, toggle vari).
- HUD overlay con stati distinti per record / hands-free / proc / AI / streaming.
- Status item del menu bar cliccabile per aprire System Settings → Privacy & Security → Accessibility quando il permesso non è concesso.

## Installazione

1. Scarica `Flow.dmg` dalla release page.
2. Apri il DMG, esegui "Install Flow.command".
3. Concedi Microphone + Accessibility quando richiesto.
4. Tieni premuto fn per dettare. Doppio fn per hands-free.

Primo avvio: l'installer scarica Python portable (~17 MB), pip install delle deps (~5 min), Whisper turbo (~1.5 GB al primo dictation), Qwen 3B (~1.8 GB al primo cleanup). Tutto offline dopo.

## Compatibilità

- macOS 13+ (Apple Silicon raccomandato)
- Funziona offline: niente cloud, niente API key
