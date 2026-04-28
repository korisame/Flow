#!/usr/bin/env python3
"""
Flow v2.0 — free, local, unlimited voice dictation for macOS
Accuracy-first · EN / IT / RU / AR · no API key · no word limits

Controls
────────
Hold Fn           push-to-talk  (release → transcribe)
Double-press Fn   hands-free    (auto-transcribes on every pause; Fn again to stop)

What's new in v2
────────────────
• Hands-free auto-segmentation — silence detection pastes each utterance live
• Verbal commands — say "period", "new line", "delete that", etc. in all 4 languages
• Recording timer — elapsed seconds shown in menu bar while recording
• Sound feedback — subtle click on start / stop
• "Paste Last" — re-paste the most recent transcription from the menu
• Full history — last 10 transcriptions kept in memory
• Model switcher — change Whisper model without restarting
• Settings saved to ~/.flow/config.json
• Launch at Login toggle
• Language-specific initial prompts for significantly better accuracy
• Hallucination suppression (no_speech_threshold, compression_ratio_threshold)
• Smart capitalisation after sentence-ending punctuation
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import queue
import webbrowser
from pathlib import Path

import numpy as np
import sounddevice as sd
import rumps
import AppKit
import Quartz

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

APP_NAME   = "Flow"
VERSION    = "1.0.0"
KOFI_URL   = "https://ko-fi.com/shaungori"

CONFIG_DIR  = Path.home() / ".flow"
CONFIG_FILE = CONFIG_DIR / "config.json"
TIMINGS_FILE = CONFIG_DIR / "timings.csv"

SAMPLE_RATE      = 16_000
BLOCK_MS         = 80          # audio chunk size in ms
HOTKEY_FLAG      = 0x800000    # kCGEventFlagMaskSecondaryFn  (Fn / Globe key)
DOUBLE_PRESS_SEC = 0.45        # double-press window (seconds)
MAX_HISTORY      = 10

# Silence detection (hands-free auto-segmentation)
SPEECH_RMS       = 0.018       # RMS level considered "speech" (tune for your mic)
SILENCE_SEC      = 1.5         # seconds of silence → auto-transcribe
MIN_UTT_SEC      = 0.4         # min utterance length worth transcribing

# ─── LANGUAGE / MODEL CONFIG ──────────────────────────────────────────────────

LANGUAGES = {
    "🌐  Auto-detect": None,
    "🇬🇧  English":    "en",
    "🇮🇹  Italiano":   "it",
    "🇫🇷  Français":   "fr",
    "🇷🇺  Русский":    "ru",
    "🇸🇦  عربي":      "ar",
}

# Seeding Whisper with a language-appropriate prompt dramatically improves
# accuracy, especially for Arabic and Russian scripts.
INITIAL_PROMPTS = {
    "en": "The following is an accurate transcription in English.",
    "it": "La seguente è una trascrizione accurata in italiano.",
    "fr": "Ce qui suit est une transcription précise en français.",
    "ru": "Ниже приведена точная транскрипция на русском языке.",
    "ar": "فيما يلي نص دقيق باللغة العربية.",
}

MODELS = [
    # (id, display label)
    ("tiny",           "tiny · 75 MB · fastest"),
    ("base",           "base · 145 MB · balanced"),
    ("small",          "small · 465 MB · good"),
    ("medium",         "medium · 1.5 GB · very good"),
    ("large-v3-turbo", "large-v3-turbo · 1.6 GB · recommended"),
    ("large-v3",       "large-v3 · 3.0 GB · most accurate"),
]

# ─── TEXT PROCESSING ──────────────────────────────────────────────────────────

FILLER_WORDS = (
    " um ", " uh ", " hmm ", " umm ", " err ",
    " um,", " uh,", " hmm,", " like like ",
)

# Verbal commands processed via regex substitution (checked in this order).
# \x00DEL is a sentinel meaning "undo the last paste".
VERBAL_CMDS = [
    # ── Delete / undo ──────────────────────────────────────────────────────────
    (r'\b(delete that|scratch that|cancel that'
     r'|cancella|annulla'
     r'|supprime ça|annule ça|efface ça'
     r'|удали|отмени'
     r'|احذف ذلك|امسح)\b',                                   '\x00DEL'),
    # ── Paragraph / line breaks ────────────────────────────────────────────────
    (r'\b(new paragraph|nuovo paragrafo|nouveau paragraphe|новый абзац|فقرة جديدة)\b', '\n\n'),
    (r'\b(new line|a capo|à la ligne|nouvelle ligne|новая строка|سطر جديد)\b',         '\n'),
    # ── Punctuation ────────────────────────────────────────────────────────────
    (r'\b(exclamation mark|punto esclamativo|point d\'exclamation'
     r'|восклицательный знак|علامة تعجب)\b',                  '!'),
    (r'\b(question mark|punto interrogativo|point d\'interrogation'
     r'|вопросительный знак|علامة استفهام)\b',                '?'),
    (r'\b(ellipsis|punti di sospensione|points de suspension|многоточие|نقاط الحذف)\b', '…'),
    (r'\b(em dash|trattino lungo|tiret cadratin|длинное тире|شرطة)\b',                  '—'),
    (r'\b(open parenthesis|apri parentesi|ouvrir parenthèse'
     r'|открыть скобку|قوس مفتوح)\b',                         '('),
    (r'\b(close parenthesis|chiudi parentesi|fermer parenthèse'
     r'|закрыть скобку|قوس مغلق)\b',                          ')'),
    (r'\b(open quote|virgolette aperte|guillemet ouvert'
     r'|открыть кавычки|فتح اقتباس)\b',                       '“'),
    (r'\b(close quote|virgolette chiuse|guillemet fermé'
     r'|закрыть кавычки|إغلاق اقتباس)\b',                     '”'),
    (r'\b(semicolon|punto e virgola|point-virgule'
     r'|точка с запятой|فاصلة منقوطة)\b',                     ';'),
    (r'\b(colon|due punti|deux points|двоеточие|نقطتان)\b',   ':'),
    # Language-specific period & comma
    # (English "period" intentionally omitted — too ambiguous in EN sentences)
    (r'\b(periodo|punto fermo)\b',                             '.'),
    (r'\b(точка)\b',                                           '.'),
    (r'\b(نقطة)\b',                                            '.'),
    (r'\b(virgola)\b',                                         ','),
    (r'\b(запятая)\b',                                         ','),
    (r'\b(فاصلة)\b',                                           ','),
]

# Compile once for speed
_VERBAL_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in VERBAL_CMDS]

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

BACKENDS = [
    # (id, display label)
    ("faster-whisper", "faster-whisper · CPU int8 · universal"),
    ("mlx-whisper",    "mlx-whisper · Apple Silicon GPU · fastest"),
]

LOCAL_LLMS = [
    # (id,            hf-repo,                                          expected_bytes, label)
    ("qwen2.5-3b",  "mlx-community/Qwen2.5-3B-Instruct-4bit",   1_800_000_000, "Qwen2.5 3B  ·  follows instructions well  (recommended)"),
    ("qwen2.5-7b",  "mlx-community/Qwen2.5-7B-Instruct-4bit",   4_200_000_000, "Qwen2.5 7B  ·  best quality / disk trade-off"),
    ("llama-3.2-3b","mlx-community/Llama-3.2-3B-Instruct-4bit", 1_700_000_000, "Llama 3.2 3B  ·  fast but creative"),
    ("llama-3.1-8b","mlx-community/Llama-3.1-8B-Instruct-4bit", 4_500_000_000, "Llama 3.1 8B  ·  solid"),
]
DEFAULT_LOCAL_LLM = "qwen2.5-3b"

def _local_llm_meta(llm_id: str) -> tuple[str, int, str]:
    """Return (hf-repo, expected_bytes, label) for a given llm id."""
    for lid, repo, size, lbl in LOCAL_LLMS:
        if lid == llm_id:
            return repo, size, lbl
    return LOCAL_LLMS[0][1], LOCAL_LLMS[0][2], LOCAL_LLMS[0][3]

def _local_llm_repo(llm_id: str) -> str:
    return _local_llm_meta(llm_id)[0]

def _local_llm_size(llm_id: str) -> int:
    return _local_llm_meta(llm_id)[1]

def _local_llm_label(llm_id: str) -> str:
    return _local_llm_meta(llm_id)[3]


def _hf_cache_dir(repo: str) -> Path:
    """~/.cache/huggingface/hub/models--<org>--<name>/"""
    return Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + repo.replace("/", "--")
    )

def _hf_cache_bytes(repo: str) -> int:
    """Total size on disk for a HF cached repo (resolves blob/snapshot links)."""
    d = _hf_cache_dir(repo)
    if not d.exists():
        return 0
    total = 0
    try:
        # `du -sk` is much faster than walking with stat() in Python for this case.
        out = subprocess.run(
            ["du", "-sk", str(d)],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            total = int(out.stdout.split()[0]) * 1024
    except Exception:
        pass
    return total

def _is_llm_downloaded(llm_id: str, threshold: float = 0.95) -> bool:
    """True if at least `threshold` × expected bytes are already on disk."""
    repo, expected, _ = _local_llm_meta(llm_id)
    actual = _hf_cache_bytes(repo)
    return actual >= int(expected * threshold)


DEFAULT_CONFIG = {
    "backend":         "mlx-whisper",   # falls back to faster-whisper if MLX unavailable
    "model":           "large-v3-turbo",
    "language":        None,
    "sound_feedback":  True,
    "remove_fillers":  True,
    "verbal_commands": True,
    "ai_backend":      "auto",          # none / local / codex / auto  (auto = codex → fallback local)
    "ai_tone":         "auto",          # auto / neutral / casual / formal / notes / code
    "llm_model":       DEFAULT_LOCAL_LLM,
    "show_hud":        True,
    "user_dictionary": [],              # list of names/terms to bias Whisper toward
}


# ─── SETTINGS I/O ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ─── RECORDER ─────────────────────────────────────────────────────────────────

class Recorder:
    """
    Thread-safe microphone recorder.

    In push-to-talk mode:  record → stop() → returns one big numpy array.
    In hands-free mode:    record continuously; on silence > SILENCE_SEC,
                           call on_utterance(audio) automatically, keeping
                           a short overlap as context for the next utterance.
    """

    def __init__(self):
        self._buf          : list         = []
        self._lock                        = threading.Lock()
        self._stream                      = None
        self._hands_free                  = False
        self._on_utterance                = None   # callback(np.ndarray)
        self._last_speech_t               = 0.0
        self._has_speech                  = False
        self._min_samples                 = int(SAMPLE_RATE * MIN_UTT_SEC)
        self._overlap_samples             = int(SAMPLE_RATE * 0.25)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, hands_free: bool = False, on_utterance=None):
        with self._lock:
            self._buf           = []
            self._hands_free    = hands_free
            self._on_utterance  = on_utterance
            self._last_speech_t = time.monotonic()
            self._has_speech    = False

        # Mic hot-swap resilience: if the default input device just changed
        # (AirPods plugged/unplugged, USB mic etc.), the previously cached
        # device list is stale → first InputStream() attempt may fail. Retry
        # once with a fresh device query before giving up.
        last_err = None
        for attempt in range(2):
            try:
                self._stream = sd.InputStream(
                    samplerate = SAMPLE_RATE,
                    channels   = 1,
                    dtype      = "float32",
                    blocksize  = int(SAMPLE_RATE * BLOCK_MS / 1000),
                    callback   = self._audio_cb,
                )
                self._stream.start()
                return
            except Exception as e:
                last_err = e
                print(f"[recorder] InputStream attempt {attempt+1} failed: {e}",
                      file=sys.stderr, flush=True)
                self._stream = None
                # Force PortAudio to refresh its device list
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception:
                    pass
                time.sleep(0.4)
        raise last_err

    def stop(self) -> np.ndarray:
        """Stop recording and return all buffered audio."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            audio      = self._concat_buf()
            self._buf  = []
        return audio

    # ── Internal ──────────────────────────────────────────────────────────────

    def _concat_buf(self) -> np.ndarray:
        if not self._buf:
            return np.array([], dtype="float32")
        return np.concatenate(self._buf).flatten()

    def _audio_cb(self, data, frames, t, status):
        chunk = data.copy()

        with self._lock:
            self._buf.append(chunk)

        # Silence detection only active in hands-free mode
        if not self._hands_free or self._on_utterance is None:
            return

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()

        if rms > SPEECH_RMS:
            self._last_speech_t = now
            self._has_speech    = True
        elif self._has_speech and (now - self._last_speech_t) > SILENCE_SEC:
            # Pause detected — flush the utterance
            self._has_speech = False
            with self._lock:
                audio     = self._concat_buf()
                # Keep a short overlap so next segment has context
                self._buf = ([audio[-self._overlap_samples:]]
                             if len(audio) > self._overlap_samples else [])

            if audio.size >= self._min_samples:
                # Fire callback on its own thread (never block the audio callback)
                threading.Thread(
                    target=self._on_utterance,
                    args=(audio,),
                    daemon=True,
                ).start()


# ─── AI CLEANUP ───────────────────────────────────────────────────────────────
#
# Optional post-processing pass over Whisper output. Two backends:
#
#   • local  → mlx-community/Llama-3.2-3B-Instruct-4bit running on Apple GPU.
#              ~1.8 GB one-time download, ~0.5–1 s latency, fully offline,
#              no rate limits, no API key.
#   • codex  → spawn OpenAI's `codex` CLI (must be authenticated with the
#              user's ChatGPT subscription via `codex login`). Uses the latest
#              models but is rate-limited by ChatGPT plan.
#
# `auto` tries codex first, falls back to local on failure (or when offline).
# `none` disables AI cleanup entirely.
#
# Tone presets shape the prompt; `auto` picks based on the frontmost app
# (Slack → casual, Mail → formal, IDE → technical, etc.).

AI_BACKENDS = [
    ("none",  "None  ·  raw Whisper output"),
    ("local", "Local LLM  ·  offline, no rate limit  (recommended)"),
    ("codex", "Codex  ·  ChatGPT subscription (online)"),
    ("auto",  "Auto  ·  Codex with local fallback"),
]

AI_TONES = [
    ("auto",     "Auto  ·  match the frontmost app"),
    ("neutral",  "Neutral"),
    ("casual",   "Casual messaging"),
    ("formal",   "Formal email"),
    ("notes",    "Clean notes"),
    ("code",     "Technical / code"),
]

class AICleanup:
    """
    Post-processes raw transcribed text.

    Public API:
        ai = AICleanup(cfg)        # respects cfg["ai_backend"], cfg["ai_tone"]
        out = ai.clean(text, app_bundle=None)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._mlx_model       = None
        self._mlx_tok         = None
        self._mlx_loaded_repo = None
        self._mlx_lock        = threading.Lock()
        self._mlx_warming     = False     # background pre-warm in flight?

    @property
    def llm_id(self) -> str:
        return self.cfg.get("llm_model", DEFAULT_LOCAL_LLM)

    def is_local_ready(self) -> bool:
        """True if the local LLM is loaded in memory OR fully cached on disk."""
        wanted_repo = _local_llm_repo(self.llm_id)
        if self._mlx_model is not None and self._mlx_loaded_repo == wanted_repo:
            return True
        return _is_llm_downloaded(self.llm_id)
        self._codex_path    = self._find_codex()

    @property
    def backend(self) -> str:
        return self.cfg.get("ai_backend", "none")

    @property
    def tone(self) -> str:
        return self.cfg.get("ai_tone", "auto")

    def is_enabled(self) -> bool:
        return self.backend != "none"

    @staticmethod
    def _find_codex() -> str | None:
        for p in ("/opt/homebrew/bin/codex", "/usr/local/bin/codex"):
            if os.path.exists(p):
                return p
        return None

    # ── Prompt building ──────────────────────────────────────────────────────

    def _effective_tone(self, app_bundle: str | None) -> str:
        if self.tone != "auto":
            return self.tone
        if not app_bundle:
            return "neutral"
        b = app_bundle.lower()
        if any(s in b for s in ("slack", "discord", "messages", "whatsapp",
                                "telegram", "instagram", "twitter")):
            return "casual"
        if any(s in b for s in ("mail", "outlook", "spark.")):
            return "formal"
        if any(s in b for s in ("xcode", "vscode", "code", "terminal", "iterm",
                                "jetbrains", "sublime", "neovim")):
            return "code"
        if any(s in b for s in ("notion", "obsidian", "bear", "craft", "ulysses")):
            return "notes"
        return "neutral"

    @staticmethod
    def _lang_name(code: str | None) -> str:
        return {
            "it": "Italian", "en": "English", "ru": "Russian", "ar": "Arabic",
            "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
            "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
        }.get((code or "").lower(), code or "the same language as the input")

    def _system_prompt(self, tone_id: str, language: str | None) -> str:
        TONE_LINES = {
            "neutral": "Use a neutral, clear tone.",
            "casual":  "Use a casual, conversational messaging tone. Short sentences. Contractions OK.",
            "formal":  "Use a formal email tone. Complete sentences. No contractions.",
            "notes":   "Format as clean prose suitable for notes. Keep it concise.",
            "code":    "Treat the input as technical writing. Preserve technical terms exactly. No marketing fluff.",
        }
        lang_name = self._lang_name(language)
        return (
            "You are a voice-dictation cleanup assistant.\n"
            f"The input is in {lang_name}. "
            f"You MUST output in {lang_name}. "
            "ABSOLUTE RULE: never translate. If the input is in Italian, output Italian. "
            "If the input is in Russian, output Russian. Foreign words quoted inside the "
            "input stay in their original form — do NOT translate them or use them as a "
            "reason to switch language.\n"
            "Tasks:\n"
            "  • Fix punctuation and capitalization.\n"
            "  • Remove filler words (\"um\", \"uh\", \"like\", \"ehm\", \"cioè\").\n"
            "  • Fix obvious word errors.\n"
            "  • Keep the original meaning.\n"
            "PUNCTUATION RULES (very important):\n"
            "  • Use periods (.) and commas (,) by default.\n"
            "  • Use exclamation marks (!) ONLY when the speaker clearly shouted, "
            "exclaimed, or used emphatic intonation. Plain statements end with a period, "
            "NOT an exclamation mark.\n"
            "  • Use question marks (?) only for actual questions.\n"
            "  • Do NOT use em-dashes (—) or en-dashes (–). Use commas, periods, "
            "or parentheses instead.\n"
            "  • Preserve proper-noun capitalization for product names and brands "
            "(\"Flow\", \"Llama\", \"iPhone\").\n"
            "Output ONLY the cleaned text. No preamble. No commentary. No quotes. "
            "No explanations. Just the cleaned sentence(s).\n"
            + TONE_LINES.get(tone_id, TONE_LINES["neutral"])
        )

    @staticmethod
    def _tone_down_punctuation(text: str) -> str:
        """
        Llama 3.2 3B is enthusiastic with exclamation marks. The system prompt
        already says "use periods", but as a belt-and-braces measure: keep at
        most ONE '!' in the output, downgrade the rest to periods. Also collapse
        runs ("!!", "!!!" → "!").
        """
        if not text:
            return text
        # Collapse repeats first
        text = re.sub(r"!+", "!", text)
        # Now keep only the first '!' — replace subsequent ones with '.'
        first = text.find("!")
        if first == -1 or text.count("!") <= 1:
            return text
        before = text[: first + 1]
        after  = text[first + 1 :].replace("!", ".")
        return before + after

    @staticmethod
    def _looks_invented(original: str, cleaned: str) -> bool:
        """
        True if `cleaned` looks like newly-generated content rather than a
        cleanup of `original`. Uses word-level Jaccard-ish overlap: a real
        cleanup re-uses ≥70 % of the input's content words.

        Rationale: small LLMs sometimes interpret instruction-shaped inputs
        ("scrivimi le cose che mancano") as a request to produce content
        instead of a request to clean. The output then shares almost no words
        with the input — a strong signal to discard.
        """
        if not cleaned or not original:
            return False
        # Token-level lowercase, strip basic punctuation
        def toks(s: str) -> set:
            cleaned_tokens = re.findall(r"[\w'’]+", s.lower())
            # Drop tiny stop-tokens; they're noise for the overlap measure.
            return {t for t in cleaned_tokens if len(t) >= 3}
        in_t  = toks(original)
        out_t = toks(cleaned)
        if not in_t or not out_t:
            return False
        # How much of the OUTPUT is grounded in the input?
        overlap = len(in_t & out_t) / len(out_t)
        return overlap < 0.70

    @staticmethod
    def _looks_translated(original: str, cleaned: str, expected_lang: str | None) -> bool:
        """
        Quick heuristic: if the LLM ignored 'don't translate' and switched to
        English when the original was a Latin-script romance language, refuse
        the cleanup. We test for stop-word ratios.
        """
        if not cleaned or not expected_lang:
            return False
        e = expected_lang.lower()
        if e == "en":
            return False  # original was English — translation impossible

        EN_STOPS = {"the","a","an","is","are","was","were","be","been","being",
                    "have","has","had","do","does","did","will","would","could","should",
                    "and","or","but","if","so","as","at","in","on","to","of","for",
                    "from","with","without","about","this","that","these","those",
                    "you","we","they","them","he","she","it","i","my","your","our","their",
                    "not","no","yes","well","just","very","really","i'm","let's","it's",
                    "don't","doesn't","didn't","won't","can't","wouldn't","shouldn't",
                    "see","said","says","say","get","got","go","goes","went","come","came",
                    "make","made","know","knew","think","thought","want","wanted",
                    "perfectly","listening"}
        IT_STOPS = {"che", "non", "di", "il", "la", "lo", "le", "un", "una", "ho",
                    "ha", "hai", "siamo", "sono", "è", "sei", "sì", "se", "per",
                    "con", "ma", "anche", "come", "questo", "questa", "quello"}
        RU_CYRIL = sum(1 for ch in cleaned if "Ѐ" <= ch <= "ӿ")
        AR_LET   = sum(1 for ch in cleaned if "؀" <= ch <= "ۿ")

        toks = [t.lower().strip(".,!?:;\"'()") for t in cleaned.split()]
        if not toks:
            return False
        en_hits = sum(1 for t in toks if t in EN_STOPS)
        it_hits = sum(1 for t in toks if t in IT_STOPS)
        en_ratio = en_hits / len(toks)

        if e == "ru" and RU_CYRIL == 0 and len(cleaned) > 10:
            return True
        if e == "ar" and AR_LET == 0 and len(cleaned) > 10:
            return True
        if e == "it" and en_ratio > 0.18 and en_hits > it_hits:
            return True
        if e in ("es", "fr", "pt", "de") and en_ratio > 0.25:
            return True
        return False

    # ── Public entry point ───────────────────────────────────────────────────

    def clean(self, text: str, app_bundle: str | None = None,
              language: str | None = None) -> str:
        text = (text or "").strip()
        if not text or not self.is_enabled():
            return text
        # Don't bother for very short inputs — nothing to clean.
        if len(text.split()) < 3:
            return text

        tone = self._effective_tone(app_bundle)
        sys_prompt = self._system_prompt(tone, language)
        backend = self.backend

        try:
            cleaned = None
            if backend == "local":
                if not self.is_local_ready():
                    print("[ai] local LLM not ready (downloading or loading) — "
                          "skipping cleanup", file=sys.stderr, flush=True)
                    return text
                cleaned = self._clean_local(sys_prompt, text, language=language)
            elif backend == "codex":
                # Codex is high-quality but slow (typically 10–20 s with
                # gpt-5.5 over a ChatGPT subscription). Give it room.
                cleaned = self._clean_codex(sys_prompt, text, timeout=25)
            elif backend == "auto":
                # Auto: prefer Codex for quality (longer timeout), fall back
                # to local LLM only if it's already cached.
                cleaned = self._clean_codex(sys_prompt, text, timeout=15)
                if not cleaned and self.is_local_ready():
                    cleaned = self._clean_local(sys_prompt, text, language=language)

            if cleaned and self._looks_translated(text, cleaned, language):
                print(f"[ai] cleanup translated to wrong language "
                      f"(expected {language}) — discarding", file=sys.stderr, flush=True)
                return text
            if cleaned and self._looks_invented(text, cleaned):
                print(f"[ai] cleanup invented content (word-overlap < 70%) "
                      f"— discarding", file=sys.stderr, flush=True)
                return text
            if cleaned:
                cleaned = self._tone_down_punctuation(cleaned)
            # _strip_meta returns "" when the model went off the rails entirely.
            if not cleaned:
                print(f"[ai] cleanup output rejected (too inflated) — "
                      f"using raw transcript", file=sys.stderr, flush=True)
                return text
            return cleaned
        except Exception as e:
            print(f"[ai] cleanup failed ({backend}): {e}", file=sys.stderr, flush=True)
        return text

    # ── Few-shot examples per language (teach the model the format) ──────────
    _FEWSHOT = {
        "it": [
            ("ehm allora oggi vorrei comprare un iphone ma non so se prendere il pro o il pro max",
             "Allora, oggi vorrei comprare un iPhone, ma non so se prendere il Pro o il Pro Max."),
            ("perfetto fatto benissimo l'app funziona benissimo sono al 90 percento",
             "Perfetto, fatto benissimo. L'app funziona benissimo, sono al 90%."),
        ],
        "en": [
            ("um so I was thinking about uh the project we discussed yesterday",
             "So I was thinking about the project we discussed yesterday."),
            ("hey can you send me that file please",
             "Hey, can you send me that file, please?"),
        ],
        "fr": [
            ("euh alors aujourd'hui je voudrais acheter un nouveau iphone mais je sais pas si prendre le pro ou le pro max",
             "Alors, aujourd'hui je voudrais acheter un nouvel iPhone, mais je ne sais pas si prendre le Pro ou le Pro Max."),
            ("ouais bah voilà l'application marche super bien on est à 90 pourcent",
             "Voilà, l'application marche super bien, on est à 90%."),
        ],
        "ru": [
            ("ну э я думаю что нужно сделать это завтра",
             "Я думаю, что нужно сделать это завтра."),
        ],
    }

    @classmethod
    def _build_messages(cls, sys_prompt: str, text: str, lang: str | None) -> list[dict]:
        """Build a few-shot chat history teaching the model the desired format."""
        msgs = [{"role": "system", "content": sys_prompt}]
        examples = cls._FEWSHOT.get((lang or "").lower(), cls._FEWSHOT["en"])
        for raw, clean in examples:
            msgs.append({"role": "user",      "content": raw})
            msgs.append({"role": "assistant", "content": clean})
        msgs.append({"role": "user", "content": text})
        return msgs

    # ── Meta-commentary stripper ─────────────────────────────────────────────

    @staticmethod
    def _strip_meta(out: str, original: str) -> str:
        """
        Aggressive scrubber for small-LLM hallucinations. Layers:
          1. Strip leading meta-headers ("Ecco il testo pulito:", "Here is the…")
          2. Unwrap surrounding quotes
          3. Cut at trailing meta-commentary openers
          4. If still drastically longer than input, keep only the first
             non-empty paragraph
        """
        if not out:
            return out
        s = out.strip()

        # 1. Strip LEADING meta-headers (single line at top), e.g.:
        #    "Ecco il testo pulito e corretto:"
        #    "Here is the cleaned text:"
        LEADING_HEADERS = re.compile(
            r"^\s*(?:"
            r"Ecco (?:il |la |un)?[^\n]{0,80}?[:.]\s*\n+|"
            r"Here(?:'s| is)\s+(?:the\s+)?[^\n]{0,80}?[:.]\s*\n+|"
            r"Cleaned(?: text)?\s*:\s*\n*|"
            r"Corrected(?: text)?\s*:\s*\n*|"
            r"Output\s*:\s*\n*|"
            r"Risultato\s*:\s*\n*"
            r")",
            flags=re.IGNORECASE,
        )
        for _ in range(3):  # peel up to 3 nested headers
            new_s = LEADING_HEADERS.sub("", s, count=1).strip()
            if new_s == s:
                break
            s = new_s

        # 2. Unwrap matching surrounding quotes (double or single, curly or straight)
        QUOTE_PAIRS = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"),
                       ("«", "»"), ("„", "“")]
        for ql, qr in QUOTE_PAIRS:
            if s.startswith(ql) and s.endswith(qr) and len(s) > 2:
                s = s[len(ql):-len(qr)].strip()
                break

        # 3. Cut on TRAILING meta-commentary openers (anywhere in the string,
        # not just after a newline — small models sometimes cram them inline).
        TRAILING_PATTERNS = [
            r"(?:\n|\.\s+|^)\s*"
            r"(Ho corretto|Ho rimosso|Ho aggiunto|Ho cambiato|Ho modificato|Ho anche|"
            r"I have corrected|I corrected|I removed|I added|I changed|I rewrote|"
            r"I also |I've |"
            r"Here(?:'s| is)\s+(?:the\s+)?(cleaned|corrected|fixed|updated|revised)|"
            r"Cleaned text:|Corrected text:|Original:|Note:|Translation:|"
            r"Spiegazione:|Modifiche:|Cambiamenti:|Esempio:|Nota:|"
            r"In summary[,:]|To summarize[,:]|"
            r"\(Note|\(Nota|\[Note|\[Nota)",
        ]
        for pat in TRAILING_PATTERNS:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                s = s[: m.start()].strip()

        # 4. Re-unwrap quotes after stripping (in case headers hid them)
        for ql, qr in QUOTE_PAIRS:
            if s.startswith(ql) and s.endswith(qr) and len(s) > 2:
                s = s[len(ql):-len(qr)].strip()
                break

        # 5. Length sanity: cleanup is meant to *clean*, not generate. Anything
        # noticeably longer than input is the model inventing content. We use
        # 1.6× as the hard ceiling — punctuation + capitalization typically
        # adds <20 % chars; 60 % headroom is generous for filler-removal cases
        # where output ends up shorter than input.
        if len(s) > max(80, int(len(original) * 1.6)):
            first_para = s.split("\n\n", 1)[0].strip()
            if 0 < len(first_para) <= int(len(original) * 1.6):
                s = first_para
            else:
                # Model went off the rails entirely — signal caller to fallback
                # to the raw (un-cleaned) text by returning empty.
                return ""

        return s.strip()

    # ── Local MLX backend ────────────────────────────────────────────────────

    def _clean_local(self, sys_prompt: str, text: str, language: str | None = None) -> str:
        # Resolve the requested LLM id from config (lazy-load on first use,
        # reload if the user picked a different model in the menu).
        wanted_id   = self.cfg.get("llm_model", DEFAULT_LOCAL_LLM)
        wanted_repo = _local_llm_repo(wanted_id)
        with self._mlx_lock:
            if self._mlx_model is None or self._mlx_loaded_repo != wanted_repo:
                print(f"[ai-local] loading {wanted_repo}…", flush=True)
                from mlx_lm import load
                self._mlx_model, self._mlx_tok = load(wanted_repo)
                self._mlx_loaded_repo = wanted_repo
                print("[ai-local] ready", flush=True)

            from mlx_lm import generate
            messages = self._build_messages(sys_prompt, text, language)
            prompt = self._mlx_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # Tight max_tokens (~1.5× input) keeps the model from drifting
            # into self-commentary.
            max_t = max(80, min(400, int(len(text) * 1.5)))
            t0 = time.time()
            out = generate(
                self._mlx_model, self._mlx_tok,
                prompt     = prompt,
                max_tokens = max_t,
                verbose    = False,
            )
            out = (out or "").strip()
            out = self._strip_meta(out, text)
            print(f"[ai-local] {time.time()-t0:.2f}s ({len(out)} chars)", flush=True)
            return out

    # ── Codex CLI backend ────────────────────────────────────────────────────

    def _clean_codex(self, sys_prompt: str, text: str, timeout: float = 8) -> str | None:
        if not self._codex_path:
            return None
        out_file = "/tmp/.flow_codex_out"
        try:
            t0 = time.time()
            r = subprocess.run(
                [self._codex_path, "exec", "--skip-git-repo-check",
                 "-o", out_file, sys_prompt],
                input          = text,
                capture_output = True,
                text           = True,
                timeout        = timeout,
            )
            print(f"[ai-codex] {time.time()-t0:.2f}s rc={r.returncode}", flush=True)
            if r.returncode == 0 and os.path.exists(out_file):
                with open(out_file) as f:
                    return self._strip_meta(f.read().strip(), text)
        except subprocess.TimeoutExpired:
            print("[ai-codex] timeout", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[ai-codex] {e}", file=sys.stderr, flush=True)
        return None


# ─── HUD OVERLAY ──────────────────────────────────────────────────────────────
#
# Floating non-activating panel that shows recording / processing state right
# next to the cursor — far more visible than the menu bar icon for users with
# eyes-on-app focus. Uses NSPanel + NSVisualEffectView for native macOS look.

class HUD:
    """Floating cursor-anchored status pill."""

    # State → (SF Symbol, label, palette color or None)
    _STATES = {
        "rec":         ("mic.fill",         "Listening…",     ("system", "red")),
        "rec_hf":      ("waveform",         "Hands-free",     ("system", "red")),
        "proc":        ("waveform",         "Transcribing…",  None),
        "ai":          ("sparkles",         "Cleaning up…",   None),
        "downloading": ("arrow.down.circle","Downloading…",   None),
    }

    def __init__(self):
        self._panel  = None
        self._label  = None
        self._icon_v = None
        self._build()

    def _build(self):
        try:
            from Foundation import NSMakeRect, NSMakePoint
            from AppKit import (
                NSPanel, NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
                NSScreenSaverWindowLevel, NSColor, NSVisualEffectView,
                NSVisualEffectMaterialHUDWindow, NSVisualEffectStateActive,
                NSImageView, NSTextField, NSFont, NSImage,
                NSImageSymbolConfiguration, NSFontWeightMedium,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorTransient,
            )

            W, H = 180, 44
            rect = NSMakeRect(0, 0, W, H)
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(NSScreenSaverWindowLevel)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setMovableByWindowBackground_(False)
            panel.setHidesOnDeactivate_(False)
            panel.setReleasedWhenClosed_(False)
            panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorTransient
            )

            # Vibrant background pill (Tahoe-style)
            blur = NSVisualEffectView.alloc().initWithFrame_(rect)
            blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
            blur.setState_(NSVisualEffectStateActive)
            blur.setWantsLayer_(True)
            blur.layer().setCornerRadius_(H / 2.0)
            blur.layer().setMasksToBounds_(True)
            panel.contentView().addSubview_(blur)
            blur.setFrame_(rect)
            blur.setAutoresizingMask_(0x12)  # width+height resizable

            # Icon
            icon_v = NSImageView.alloc().initWithFrame_(NSMakeRect(14, 10, 24, 24))
            icon_v.setImageScaling_(2)        # NSImageScaleProportionallyUpOrDown
            blur.addSubview_(icon_v)

            # Label
            lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(46, 12, W - 56, 20))
            lbl.setBezeled_(False)
            lbl.setDrawsBackground_(False)
            lbl.setEditable_(False)
            lbl.setSelectable_(False)
            lbl.setFont_(NSFont.systemFontOfSize_weight_(13.0, NSFontWeightMedium))
            lbl.setTextColor_(NSColor.labelColor())
            lbl.setStringValue_("")
            blur.addSubview_(lbl)

            self._panel  = panel
            self._label  = lbl
            self._icon_v = icon_v
        except Exception as e:
            print(f"[hud] init failed: {e}", file=sys.stderr, flush=True)
            self._panel = None

    @staticmethod
    def _make_symbol(name: str, color_kind=None):
        try:
            img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                name, None
            )
            if img is None:
                return None
            cfg = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                16.0, 0.0
            )
            if color_kind == ("system", "red"):
                pcfg = AppKit.NSImageSymbolConfiguration.configurationWithPaletteColors_(
                    [AppKit.NSColor.systemRedColor()]
                )
                cfg = cfg.configurationByApplyingConfiguration_(pcfg)
            img = img.imageWithSymbolConfiguration_(cfg)
            img.setTemplate_(color_kind is None)
            return img
        except Exception:
            return None

    def _position_near_cursor(self):
        """Place the panel just below-right of the current mouse position."""
        try:
            loc = AppKit.NSEvent.mouseLocation()       # screen coords, bottom-left origin
            frame = self._panel.frame()
            x = loc.x + 18
            y = loc.y - frame.size.height - 12
            # Clamp to main screen
            screen = AppKit.NSScreen.mainScreen()
            if screen:
                vf = screen.visibleFrame()
                x = min(max(x, vf.origin.x + 8),
                        vf.origin.x + vf.size.width  - frame.size.width  - 8)
                y = min(max(y, vf.origin.y + 8),
                        vf.origin.y + vf.size.height - frame.size.height - 8)
            self._panel.setFrameOrigin_(AppKit.NSMakePoint(x, y))
        except Exception:
            pass

    def show(self, state: str, label_override: str | None = None):
        """Show HUD with a state preset; optionally override the label text
        (e.g. to inject a download percentage)."""
        if self._panel is None:
            return
        spec = self._STATES.get(state)
        if not spec:
            return
        sym, default_label, color = spec
        label = label_override if label_override is not None else default_label

        def _do():
            img = self._make_symbol(sym, color)
            if img is not None:
                self._icon_v.setImage_(img)
            self._label.setStringValue_(label)
            # Position only on first show (so percentage updates don't keep
            # snapping the panel to the cursor mid-drag).
            if not self._panel.isVisible():
                self._position_near_cursor()
            self._panel.orderFrontRegardless()

        if AppKit.NSThread.isMainThread():
            _do()
        else:
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_do)

    def update_label(self, text: str):
        """Update only the text label (no icon/position change)."""
        if self._panel is None:
            return
        def _do():
            self._label.setStringValue_(text)
        if AppKit.NSThread.isMainThread():
            _do()
        else:
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_do)

    def hide(self):
        if self._panel is None:
            return
        def _do():
            self._panel.orderOut_(None)
        if AppKit.NSThread.isMainThread():
            _do()
        else:
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_do)


# ─── MAIN APP ─────────────────────────────────────────────────────────────────

class FlowApp(rumps.App):

    # Menu-bar icon states. Each state maps to an SF Symbol name + optional
    # palette color. We render them to TIFF on disk at startup and switch by
    # setting `self.icon = path` (rumps-compatible). Fallback to text emoji
    # if the symbol is unavailable on this macOS release.
    _STATE_LOADING = "loading"
    _STATE_IDLE    = "idle"
    _STATE_REC     = "rec"
    _STATE_REC_HF  = "rec_hf"
    _STATE_PROC    = "proc"

    # Fallback strings (only used if SF Symbol rendering fails)
    _FALLBACK_TEXT = {
        _STATE_LOADING: "⏳",
        _STATE_IDLE:    "🎙",
        _STATE_REC:     "🔴",
        _STATE_REC_HF:  "🔴∞",
        _STATE_PROC:    "💭",
    }

    def __init__(self):
        # Pre-render SF Symbol icons before calling rumps init so we can pass
        # the loading icon path straight to the status item.
        self._icons = self._render_state_icons()

        # rumps.App requires either a title or an icon. We pass title=" "
        # (single space) and immediately replace via `self.icon` if available.
        super().__init__(" ", quit_button=None)
        self._apply_state(self._STATE_LOADING)

        # Config & model state
        self._cfg          = load_config()
        self._model        = None
        self._model_name   = self._cfg["model"]
        self._backend      = self._cfg.get("backend", "faster-whisper")
        self._model_lock   = threading.Lock()

        # Recording state
        self._recorder     = Recorder()
        self._language     = self._cfg["language"]
        self._q            = queue.Queue()
        self._recording    = False
        self._hands_free   = False
        self._fn_down      = False
        self._last_fn      = 0.0
        self._rec_start    = 0.0
        self._stop_timer   = threading.Event()

        # History  (newest first)
        self._history: list[str] = []

        # AI cleanup + HUD overlay
        self._ai  = AICleanup(self._cfg)
        self._hud = HUD()

        # State for the LLM download progress UI
        self._llm_dl_lock      = threading.Lock()
        self._llm_dl_in_flight = False

        # Build UI
        self._build_menu()

        # Background threads
        threading.Thread(target=self._load_model,        daemon=True).start()
        threading.Thread(target=self._transcribe_worker, daemon=True).start()
        threading.Thread(target=self._event_tap,         daemon=True).start()
        # Pre-warm the local LLM in the background so the first dictation
        # doesn't have to wait for ~1.8 GB to come down from the HF Hub.
        threading.Thread(target=self._prewarm_local_llm, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # NATIVE STATUS-BAR ICONS  (SF Symbols → TIFF on disk → rumps `icon` path)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _render_state_icons() -> dict:
        """
        Render an SF Symbol per state to ~/.flow/icons/<state>.tiff and return
        a dict {state: path or None}. Falls back to None silently when the
        symbol API is unavailable (pre-Big Sur) or rendering fails.
        """
        try:
            ic_dir = Path.home() / ".flow" / "icons"
            ic_dir.mkdir(parents=True, exist_ok=True)

            POINT_SIZE = 16.0
            try:
                font_weight = AppKit.NSFontWeightRegular
            except AttributeError:
                font_weight = 0.0  # NSFontWeightRegular ≈ 0.0

            red = AppKit.NSColor.systemRedColor()

            specs = {
                "loading": ("circle.dotted",       None),
                "idle":    ("mic.fill",            None),
                "rec":     ("mic.fill",            red),
                "rec_hf":  ("waveform",            red),
                "proc":    ("waveform",            None),
            }

            out = {}
            for state, (sym_name, palette) in specs.items():
                img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                    sym_name, f"Flow {state}"
                )
                if img is None:
                    out[state] = None
                    continue

                cfg = AppKit.NSImageSymbolConfiguration\
                    .configurationWithPointSize_weight_(POINT_SIZE, font_weight)
                if palette is not None:
                    pcfg = AppKit.NSImageSymbolConfiguration\
                        .configurationWithPaletteColors_([palette])
                    cfg = cfg.configurationByApplyingConfiguration_(pcfg)
                img = img.imageWithSymbolConfiguration_(cfg)
                # Template = adapts to dark/light system tint when no palette.
                img.setTemplate_(palette is None)

                tiff = img.TIFFRepresentation()
                if tiff is None:
                    out[state] = None
                    continue
                path = str(ic_dir / f"{state}.tiff")
                tiff.writeToFile_atomically_(path, True)
                out[state] = path
            return out
        except Exception as e:
            print(f"[icons] render failed: {e}", flush=True)
            return {}

    @staticmethod
    def _on_main(fn):
        """
        Run `fn` on the AppKit main thread. macOS Tahoe (and recent macOS
        releases in general) crash hard with "Must only be used from the main
        thread" if any AppKit method (NSMenuItem.setTitle:, NSImage swap on the
        status item, etc.) is called from a worker thread. Every UI mutation
        in Flow goes through this helper.
        """
        if AppKit.NSThread.isMainThread():
            fn()
        else:
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(fn)

    def _set_menu_title(self, item, title: str):
        """Thread-safe NSMenuItem title setter."""
        self._on_main(lambda: setattr(item, "title", title))

    def _apply_state(self, state: str, suffix: str = ""):
        """
        Update the menu-bar status item to show the icon (and optional text
        suffix, e.g. recording timer). Always dispatched to the main thread.
        Falls back to emoji text if the icon for this state isn't available.
        """
        icon_path = self._icons.get(state)

        def update():
            if icon_path:
                self.icon  = icon_path
                self.title = suffix          # e.g. " 5s" while recording
            else:
                # No SF Symbol available — fall back to legacy emoji
                self.icon  = None
                text = self._FALLBACK_TEXT.get(state, "")
                self.title = f"{text} {suffix}".rstrip()
        self._on_main(update)

    # ══════════════════════════════════════════════════════════════════════════
    # MENU
    # ══════════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        # ── Language submenu ──────────────────────────────────────────────────
        self._lang_items: dict[str, rumps.MenuItem] = {}
        lang_menu = rumps.MenuItem("Language")
        for label, code in LANGUAGES.items():
            item = rumps.MenuItem(label, callback=self._cb_set_language)
            item.state = (code == self._language or
                          (code is None and self._language is None))
            self._lang_items[label] = item
            lang_menu.add(item)

        # ── Model submenu ─────────────────────────────────────────────────────
        # Native checkmark from rumps (item.state) marks the active row — no
        # manual ✓ in the description, no leading-spaces hack.
        self._model_items: dict[str, rumps.MenuItem] = {}
        model_menu = rumps.MenuItem("Whisper Model")
        for name, desc in MODELS:
            item             = rumps.MenuItem(desc, callback=self._cb_set_model)
            item._flow_model = name
            item.state       = (name == self._model_name)
            self._model_items[name] = item
            model_menu.add(item)

        # ── Backend submenu ───────────────────────────────────────────────────
        self._backend_items: dict[str, rumps.MenuItem] = {}
        backend_menu = rumps.MenuItem("Backend")
        for name, desc in BACKENDS:
            item               = rumps.MenuItem(desc, callback=self._cb_set_backend)
            item._flow_backend = name
            item.state         = (name == self._backend)
            self._backend_items[name] = item
            backend_menu.add(item)

        # ── AI cleanup backend submenu ────────────────────────────────────────
        self._ai_backend_items: dict[str, rumps.MenuItem] = {}
        ai_menu = rumps.MenuItem("AI Cleanup")
        for name, desc in AI_BACKENDS:
            item                 = rumps.MenuItem(desc, callback=self._cb_set_ai_backend)
            item._flow_ai_backend = name
            item.state           = (name == self._cfg.get("ai_backend", "none"))
            self._ai_backend_items[name] = item
            ai_menu.add(item)

        # ── AI tone submenu ───────────────────────────────────────────────────
        self._ai_tone_items: dict[str, rumps.MenuItem] = {}
        tone_menu = rumps.MenuItem("Cleanup Tone")
        for name, desc in AI_TONES:
            item              = rumps.MenuItem(desc, callback=self._cb_set_ai_tone)
            item._flow_ai_tone = name
            item.state        = (name == self._cfg.get("ai_tone", "auto"))
            self._ai_tone_items[name] = item
            tone_menu.add(item)

        # ── Local LLM model submenu ───────────────────────────────────────────
        self._llm_items: dict[str, rumps.MenuItem] = {}
        llm_menu = rumps.MenuItem("Local LLM")
        cur_llm = self._cfg.get("llm_model", DEFAULT_LOCAL_LLM)
        for lid, _repo, expected_bytes, label in LOCAL_LLMS:
            disk_gb = expected_bytes / 1_000_000_000
            item             = rumps.MenuItem(f"{label}  ({disk_gb:.1f} GB)", callback=self._cb_set_llm)
            item._flow_llm   = lid
            item.state       = (lid == cur_llm)
            self._llm_items[lid] = item
            llm_menu.add(item)

        # ── Toggles ───────────────────────────────────────────────────────────
        self._snd_item  = rumps.MenuItem("Sound Feedback",       callback=self._cb_toggle_sound)
        self._snd_item.state  = self._cfg["sound_feedback"]

        self._fil_item  = rumps.MenuItem("Remove Filler Words",  callback=self._cb_toggle_fillers)
        self._fil_item.state  = self._cfg["remove_fillers"]

        self._cmd_item  = rumps.MenuItem("Verbal Commands",      callback=self._cb_toggle_commands)
        self._cmd_item.state  = self._cfg["verbal_commands"]

        self._hud_item  = rumps.MenuItem("Show HUD overlay",     callback=self._cb_toggle_hud)
        self._hud_item.state = self._cfg.get("show_hud", True)

        self._login_item = rumps.MenuItem("Launch at Login",     callback=self._cb_toggle_login)
        self._login_item.state = self._check_login_item()

        # ── Dynamic items ─────────────────────────────────────────────────────
        self._status_item = rumps.MenuItem("Loading model…", callback=self._cb_status_clicked)
        self._last_item   = rumps.MenuItem("Last paste — none yet", callback=self._cb_paste_last)

        # ── Assemble — native Mac structure: header → shortcuts → settings →
        # last paste → toggles → status → about ────────────────────────────
        self.menu = [
            rumps.MenuItem(f"Flow {VERSION}"),
            None,
            rumps.MenuItem("Hold Fn — push to talk"),
            rumps.MenuItem("Press Fn twice — hands-free"),
            None,
            lang_menu,
            model_menu,
            backend_menu,
            None,
            ai_menu,
            tone_menu,
            llm_menu,
            rumps.MenuItem("Edit Dictionary…", callback=self._cb_edit_dictionary),
            None,
            self._last_item,
            None,
            self._snd_item,
            self._fil_item,
            self._cmd_item,
            self._hud_item,
            self._login_item,
            None,
            self._status_item,
            None,
            rumps.MenuItem("Support on Ko-fi…", callback=lambda _: webbrowser.open(KOFI_URL)),
            None,
            rumps.MenuItem("Quit Flow", callback=lambda _: rumps.quit_application()),
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # MENU CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _cb_set_language(self, sender):
        for item in self._lang_items.values():
            item.state = False
        sender.state = True
        for label, code in LANGUAGES.items():
            if label == sender.title:
                self._language = code
                self._cfg["language"] = code
                save_config(self._cfg)
                break

    def _cb_set_model(self, sender):
        name = sender._flow_model
        if name == self._model_name:
            return
        for item in self._model_items.values():
            item.state = False
        sender.state        = True
        self._model_name    = name
        self._cfg["model"]  = name
        save_config(self._cfg)
        # Reload the model in background
        with self._model_lock:
            self._model = None
        self._set_menu_title(self._status_item, f"Loading {name}…")
        self._apply_state(self._STATE_LOADING)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _cb_set_backend(self, sender):
        name = sender._flow_backend
        if name == self._backend:
            return
        for item in self._backend_items.values():
            item.state = False
        sender.state         = True
        self._backend        = name
        self._cfg["backend"] = name
        save_config(self._cfg)
        with self._model_lock:
            self._model = None
        self._set_menu_title(self._status_item, f"Switching backend to {name}…")
        self._apply_state(self._STATE_LOADING)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _cb_toggle_sound(self, sender):
        sender.state               = not sender.state
        self._cfg["sound_feedback"] = bool(sender.state)
        save_config(self._cfg)

    def _cb_toggle_fillers(self, sender):
        sender.state               = not sender.state
        self._cfg["remove_fillers"] = bool(sender.state)
        save_config(self._cfg)

    def _cb_toggle_commands(self, sender):
        sender.state                = not sender.state
        self._cfg["verbal_commands"] = bool(sender.state)
        save_config(self._cfg)

    def _cb_toggle_hud(self, sender):
        sender.state           = not sender.state
        self._cfg["show_hud"]  = bool(sender.state)
        save_config(self._cfg)
        if not sender.state:
            self._hud.hide()

    def _cb_set_ai_backend(self, sender):
        name = sender._flow_ai_backend
        if name == self._cfg.get("ai_backend"):
            return
        for item in self._ai_backend_items.values():
            item.state = False
        sender.state             = True
        self._cfg["ai_backend"]  = name
        save_config(self._cfg)
        # AICleanup reads from self.cfg every call, so no reload needed.

    def _cb_set_ai_tone(self, sender):
        name = sender._flow_ai_tone
        if name == self._cfg.get("ai_tone"):
            return
        for item in self._ai_tone_items.values():
            item.state = False
        sender.state          = True
        self._cfg["ai_tone"]  = name
        save_config(self._cfg)

    def _cb_set_llm(self, sender):
        lid = sender._flow_llm
        if lid == self._cfg.get("llm_model"):
            return
        for item in self._llm_items.values():
            item.state = False
        sender.state          = True
        self._cfg["llm_model"] = lid
        save_config(self._cfg)
        # Drop the currently-loaded LLM so the new selection takes effect.
        with self._ai._mlx_lock:
            self._ai._mlx_model       = None
            self._ai._mlx_tok         = None
            self._ai._mlx_loaded_repo = None
        # Kick off the pre-warm (downloads if needed, with HUD + notification).
        threading.Thread(target=self._prewarm_local_llm, daemon=True).start()

    def _cb_edit_dictionary(self, _):
        """
        Open a small text dialog where the user can list important nouns/names
        (one per line or comma-separated) that Whisper otherwise mis-transcribes.
        Stored in cfg['user_dictionary']; injected into Whisper's initial_prompt
        on every transcription.
        """
        current = self._cfg.get("user_dictionary") or []
        default_text = "\n".join(current) if current else "Llama\nFlow\nGioele"
        win = rumps.Window(
            title         = "User Dictionary",
            message       = ("Words and names Whisper should recognize better.\n"
                             "One per line. Examples: brand names, contacts, "
                             "technical terms."),
            default_text  = default_text,
            ok            = "Save",
            cancel        = "Cancel",
            dimensions    = (380, 180),
        )
        try:
            response = win.run()
        except Exception as e:
            print(f"[dict] window error: {e}", file=sys.stderr, flush=True)
            return
        if response.clicked != 1:
            return
        words = []
        for line in response.text.splitlines():
            for tok in re.split(r"[,\n;]", line):
                tok = tok.strip()
                if tok and tok not in words:
                    words.append(tok)
        self._cfg["user_dictionary"] = words
        save_config(self._cfg)
        rumps.notification(
            "Flow",
            "Dictionary updated",
            f"{len(words)} terms — applied on next dictation.",
        )

    def _cb_toggle_login(self, sender):
        path = self._app_path()
        if sender.state:
            script = 'tell application "System Events" to delete (login items where name is "Flow")'
        else:
            script = (
                'tell application "System Events" to make new login item at end '
                f'of login items with properties {{path:"{path}", hidden:false}}'
            )
        subprocess.run(["osascript", "-e", script], capture_output=True)
        sender.state = not sender.state

    def _cb_status_clicked(self, sender):
        """
        Make the status menu item actionable when it surfaces a permission
        problem. Click → jump straight to the matching System Settings pane.
        Otherwise it's a no-op (just informational).
        """
        title = (sender.title or "").lower()
        if "accessibility" in title:
            subprocess.Popen([
                "open",
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
            ])
        elif "microphone" in title:
            subprocess.Popen([
                "open",
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
            ])
        # else: leave it alone (model loading / ready / error are passive)

    def _cb_paste_last(self, _):
        if self._history:
            self._paste(self._history[0])

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL
    # ══════════════════════════════════════════════════════════════════════════

    def _load_model(self):
        name    = self._model_name        # capture at start (may change)
        backend = self._backend
        try:
            if backend == "mlx-whisper":
                # MLX backend: model is just the HF repo path; no warmup load needed
                # (mlx_whisper.transcribe loads on first call and caches it)
                import mlx_whisper  # noqa — import test only
                # mlx-community uses an "-mlx" suffix for most converted
                # checkpoints; a few (notably large-v3-turbo) are published
                # without the suffix.
                NO_SUFFIX = {"large-v3-turbo"}
                repo_name = name if name in NO_SUFFIX else f"{name}-mlx"
                repo = f"mlx-community/whisper-{repo_name}"
                model = ("mlx", repo)
            else:
                from faster_whisper import WhisperModel
                model = ("fw", WhisperModel(name, device="auto", compute_type="int8"))
            with self._model_lock:
                if self._model_name == name and self._backend == backend:
                    self._model = model
            self._apply_state(self._STATE_IDLE)
            self._set_menu_title(self._status_item, f"{name} · {backend} · ready")
        except Exception as e:
            self._set_menu_title(self._status_item, f"Model error: {e}")
            rumps.notification("Flow", "Model failed to load", str(e))
            print(f"[model] load failed ({backend}/{name}): {e}", file=sys.stderr, flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL HOTKEY (CGEventTap)
    # ══════════════════════════════════════════════════════════════════════════

    def _event_tap(self):
        # Crash-recovery wrapper around the real event tap. If the run loop
        # exits (CGEventTap timeout, signal, etc.), reinstall it.
        while True:
            try:
                self._event_tap_inner()
            except Exception as e:
                import traceback
                print(f"[tap] crashed, restarting: {e}\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                time.sleep(1)
            else:
                # _event_tap_inner returned cleanly (e.g. accessibility denied).
                # Don't busy-loop — wait before retry.
                time.sleep(2)

    def _event_tap_inner(self):
        app = self
        print("[tap] event tap thread starting", flush=True)

        def _cb(proxy, etype, event, refcon):
            try:
                if etype == Quartz.kCGEventFlagsChanged:
                    flags = Quartz.CGEventGetFlags(event)
                    fn_now = bool(flags & HOTKEY_FLAG)
                    # DEBUG: log every flagsChanged event so we can see WHAT flag the Fn/Globe sends
                    print(f"[tap] flagsChanged flags=0x{flags:x} fn={fn_now}", flush=True)
                    if fn_now and not app._fn_down:
                        app._fn_down = True
                        app._on_fn_press()
                    elif not fn_now and app._fn_down:
                        app._fn_down = False
                        app._on_fn_release()
            except Exception as e:
                print(f"[tap] cb error: {e}", file=sys.stderr, flush=True)
            return event

        # Listen to BOTH flagsChanged AND keyDown so we can see if Fn comes through
        # as a separate keyDown event (some Macs deliver Globe as keyDown not flagsChanged)
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp))

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            _cb, None,
        )
        print(f"[tap] CGEventTapCreate returned: {tap!r}", flush=True)

        if tap is None:
            self._set_menu_title(self._status_item, "⚠️  Grant Accessibility permission — click to open Settings")
            rumps.notification(
                "Flow — Action Required",
                "Accessibility permission needed",
                "System Settings → Privacy & Security → Accessibility → add Flow",
            )
            return

        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(loop, src, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)
        print("[tap] event tap installed and enabled, entering CFRunLoopRun", flush=True)
        Quartz.CFRunLoopRun()
        print("[tap] CFRunLoopRun returned (unexpected!)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HOTKEY LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _on_fn_press(self):
        now = time.monotonic()

        if self._hands_free:
            # Any press while in hands-free → stop and transcribe remaining audio
            self._stop_recording(end_hf=True)
            return

        # Double-press → engage hands-free
        if now - self._last_fn < DOUBLE_PRESS_SEC:
            self._hands_free = True
            if not self._recording:
                self._start_recording()
            self._apply_state(self._STATE_REC_HF)
            return

        self._last_fn = now
        self._start_recording()

    def _on_fn_release(self):
        if self._recording and not self._hands_free:
            self._stop_recording()

    # ══════════════════════════════════════════════════════════════════════════
    # RECORDING CONTROL
    # ══════════════════════════════════════════════════════════════════════════

    def _start_recording(self):
        with self._model_lock:
            model_ready = self._model is not None
        if not model_ready:
            rumps.notification("Flow", "Still loading", "Model is loading — try again in a moment.")
            return
        if self._recording:
            return

        self._recording  = True
        self._rec_start  = time.monotonic()
        self._stop_timer.clear()

        if self._cfg["sound_feedback"]:
            try:
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Pop.aiff"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        state = self._STATE_REC_HF if self._hands_free else self._STATE_REC
        self._apply_state(state)
        if self._cfg.get("show_hud", True):
            self._hud.show("rec_hf" if self._hands_free else "rec")

        self._recorder.start(
            hands_free    = self._hands_free,
            on_utterance  = (self._enqueue_utterance if self._hands_free else None),
        )

        # Live timer thread
        threading.Thread(target=self._run_timer, daemon=True).start()

    def _stop_recording(self, end_hf: bool = False):
        if not self._recording:
            return
        self._recording = False
        self._stop_timer.set()          # kill the timer thread

        if end_hf:
            self._hands_free = False

        if self._cfg["sound_feedback"]:
            # afplay subprocess instead of AppKit.NSSound (thread-safe)
            try:
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Tink.aiff"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        self._apply_state(self._STATE_PROC)
        if self._cfg.get("show_hud", True):
            self._hud.show("proc")
        audio = self._recorder.stop()
        audio = self._trim_trailing_silence(audio)

        # Enqueue if there's meaningful audio (avoids transcribing accidental taps)
        if audio.size > int(SAMPLE_RATE * 1.0):
            self._q.put(audio)
        else:
            self._apply_state(self._STATE_IDLE)
            self._hud.hide()

    def _enqueue_utterance(self, audio: np.ndarray):
        """Called by Recorder during hands-free pause detection."""
        audio = self._trim_trailing_silence(audio)
        if audio.size > int(SAMPLE_RATE * 1.0):
            self._q.put(audio)

    @staticmethod
    def _trim_trailing_silence(
        audio: np.ndarray,
        rms_threshold: float = 0.005,
        win_ms: int = 30,
        keep_pad_ms: int = 400,
    ) -> np.ndarray:
        """
        Crop trailing silence from `audio` to stop Whisper from hallucinating
        words past the last real speech. Walks back from the end in 30-ms
        windows; the cut point is the last window whose RMS exceeds the
        threshold, plus 250 ms of padding so we don't chop the tail of the
        last word.
        """
        if audio.size == 0:
            return audio
        win = int(SAMPLE_RATE * win_ms / 1000)
        if win <= 0 or audio.size < win:
            return audio
        # Iterate from the end backwards
        i = audio.size
        last_speech_end = None
        while i - win >= 0:
            chunk = audio[i - win : i]
            rms   = float(np.sqrt(np.mean(chunk * chunk)))
            if rms > rms_threshold:
                last_speech_end = i
                break
            i -= win
        if last_speech_end is None:
            return audio          # all silent: let caller's size check drop it
        cut = min(audio.size, last_speech_end + int(SAMPLE_RATE * keep_pad_ms / 1000))
        return audio[:cut]

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _run_timer(self):
        start = time.monotonic()
        while not self._stop_timer.wait(timeout=0.5):
            elapsed = int(time.monotonic() - start)
            suffix  = "∞" if self._hands_free else ""
            state = self._STATE_REC_HF if self._hands_free else self._STATE_REC
            self._apply_state(state, suffix=f" {elapsed}s")

    # ══════════════════════════════════════════════════════════════════════════
    # TRANSCRIPTION WORKER
    # ══════════════════════════════════════════════════════════════════════════

    def _transcribe_worker(self):
        # Outer crash-recovery loop: if the inner body raises (memory error,
        # MLX hiccup, model unload, etc.) we log and resume rather than killing
        # the queue worker.
        while True:
            try:
                self._transcribe_worker_inner()
            except Exception as e:
                import traceback
                print(f"[worker] crashed, restarting: {e}\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                time.sleep(1)

    def _transcribe_worker_inner(self):
        while True:
            audio = self._q.get()
            audio_s = audio.size / SAMPLE_RATE
            print(f"[transcribe] got audio: {audio.size} samples ({audio_s:.2f}s)", flush=True)

            # Per-utterance timing breakdown — populated as we go and dumped
            # both to console and to ~/.flow/timings.csv at the end.
            t = {
                "start":        time.monotonic(),
                "audio_s":      audio_s,
                "transcribe_s": 0.0,
                "cleanup_s":    0.0,
                "paste_s":      0.0,
                "total_s":      0.0,
                "chars_in":     0,
                "chars_out":    0,
                "language":     "",
                "ai_used":      False,
                "ai_kept":      False,
            }

            # Don't overwrite the recording icon if we're still in hands-free
            if not self._recording:
                self._apply_state(self._STATE_PROC)
            try:
                with self._model_lock:
                    model = self._model
                if model is None:
                    print("[transcribe] WARNING: no model loaded yet, skipping audio", flush=True)
                    continue

                lang   = self._language
                prompt = INITIAL_PROMPTS.get(lang, "") if lang else ""

                # User dictionary: append important names/terms so Whisper is
                # biased toward recognizing them (e.g. "Llama", "iCloud",
                # personal names that would otherwise be mis-transcribed).
                user_dict = self._cfg.get("user_dictionary") or []
                if user_dict:
                    extra = " ".join(str(w) for w in user_dict if str(w).strip())
                    prompt = (prompt + " " + extra).strip() if prompt else extra

                kind, mdl = model
                ts_transcribe = time.monotonic()
                if kind == "mlx":
                    # MLX backend (Apple Silicon GPU)
                    # Tuned for long-form dictation: keep low-confidence words,
                    # don't aggressively drop quiet passages, condition on
                    # previous text for cross-30s-chunk continuity.
                    import mlx_whisper
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo            = mdl,
                        language                   = lang,
                        initial_prompt             = prompt,
                        # Temperature fallback: try greedy first, escalate if a
                        # segment fails compression / logprob checks.
                        temperature                = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        condition_on_previous_text = True,
                        compression_ratio_threshold= 2.4,
                        logprob_threshold          = -2.0,   # was -1.0 (less aggressive drop)
                        no_speech_threshold        = 0.3,    # was 0.6 (less aggressive silence cut)
                        word_timestamps            = False,
                        verbose                    = False,
                    )
                    text = (result.get("text") or "").strip()
                    detected_lang = result.get("language") or lang
                else:
                    # faster-whisper backend (CPU int8)
                    segments, info = mdl.transcribe(
                        audio,
                        language                    = lang,
                        initial_prompt              = prompt,
                        beam_size                   = 5,
                        temperature                 = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        vad_filter                  = True,
                        vad_parameters              = dict(
                            min_silence_duration_ms = 1000,  # was 200 (don't cut on short pauses)
                            speech_pad_ms           = 400,
                        ),
                        condition_on_previous_text  = True,
                        compression_ratio_threshold = 2.4,
                        log_prob_threshold          = -2.0,
                        no_speech_threshold         = 0.3,
                        word_timestamps             = False,
                    )
                    text = " ".join(seg.text for seg in segments).strip()
                    detected_lang = (info.language if info is not None else lang) or lang

                t["transcribe_s"] = time.monotonic() - ts_transcribe
                t["language"]     = detected_lang or ""
                t["chars_in"]     = len(text)
                print(f"[transcribe] raw text: {text!r}  (lang={detected_lang})", flush=True)

                # Anti-hallucination guard: Whisper sometimes emits a single
                # character repeated dozens of times (e.g. "ギギギ…" on silent
                # or near-silent audio). Detect and discard.
                if text and self._is_hallucination(text):
                    print(f"[transcribe] hallucination detected — discarding", flush=True)
                    text = ""

                text = self._process_text(text)
                print(f"[transcribe] processed text: {text!r}", flush=True)

                # AI cleanup pass (LLM): only run if enabled and the result
                # isn't a special sentinel (e.g. delete-that command).
                if (text and text != "\x00DEL"
                        and self._ai.is_enabled()):
                    if self._cfg.get("show_hud", True):
                        self._hud.show("ai")
                    app_bundle = self._frontmost_app_bundle()
                    t["ai_used"] = True
                    ts_cleanup = time.monotonic()
                    cleaned = self._ai.clean(
                        text,
                        app_bundle = app_bundle,
                        language   = detected_lang,
                    )
                    t["cleanup_s"] = time.monotonic() - ts_cleanup
                    if cleaned and cleaned != text:
                        print(f"[ai] cleaned ({len(text)} → {len(cleaned)} chars)", flush=True)
                        text = cleaned
                        t["ai_kept"] = True

                if text == "\x00DEL":
                    self._undo()
                elif text:
                    print(f"[paste] starting paste of {len(text)} chars", flush=True)
                    ts_paste = time.monotonic()
                    self._paste(text)
                    t["paste_s"]   = time.monotonic() - ts_paste
                    t["chars_out"] = len(text)
                    print(f"[paste] done", flush=True)
                    self._add_to_history(text)
                else:
                    print(f"[transcribe] empty text after processing — nothing to paste", flush=True)

            except Exception as e:
                import traceback
                print(f"[transcribe] EXCEPTION: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                rumps.notification("Flow", "Transcription error", str(e))
            finally:
                if not self._recording:
                    self._apply_state(self._STATE_IDLE)
                    self._hud.hide()
                # Emit timing summary (console line + CSV row) for every
                # utterance, regardless of success/failure.
                t["total_s"] = time.monotonic() - t["start"]
                self._emit_timing(t)

    # ══════════════════════════════════════════════════════════════════════════
    # TEXT PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def _is_hallucination(self, text: str) -> bool:
        """
        Heuristic to detect Whisper hallucinations on silent/near-silent audio,
        or cross-language confusion (e.g. transcribing Russian audio as Chinese).
        """
        s = text.strip()
        if not s:
            return False

        # 1. Stripped of whitespace, ≥10 chars and ≤3 unique characters → repetition
        compact = "".join(s.split())
        if len(compact) >= 10 and len(set(compact)) <= 3:
            return True

        # 2. Known Whisper stock hallucinations
        STOCK = (
            "thanks for watching",
            "thank you for watching",
            "subscribe to",
            "ご視聴ありがとうございました",
            "字幕志愿者",
            "amara.org",
            "iscriviti al canale",
            "grazie per aver guardato",
        )
        low = s.lower()
        for h in STOCK:
            if h in low:
                return True

        # 3. CJK-script output when target language is NOT CJK → cross-lang
        # confusion. Common failure mode: Russian/Arabic audio → Chinese chars.
        expected = self._language  # may be None for auto-detect
        if expected and expected not in ("zh", "ja", "ko"):
            cjk = sum(
                1 for ch in s
                if ('一' <= ch <= '鿿')   # CJK Unified Ideographs
                or ('぀' <= ch <= 'ヿ')   # Hiragana / Katakana
                or ('가' <= ch <= '힯')   # Hangul
            )
            if cjk >= 3 and cjk / max(1, len(compact)) > 0.2:
                return True

        # 4. Heavy repetition: same token repeated >5 times consecutively
        words = s.split()
        if len(words) >= 6:
            run = 1
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    run += 1
                    if run >= 5:
                        return True
                else:
                    run = 1

        return False

    def _process_text(self, text: str) -> str:
        if not text:
            return ""

        # 0. Normalise typographic punctuation that Whisper loves to emit but
        # which is annoying inside chat apps / code / search bars.
        #   curly quotes  → straight quotes
        #   ellipsis      → three dots
        #   NBSP          → space
        # NOTE: dashes are stripped further down (step 0b) — not mapped here.
        TYPO_MAP = {
            "“": '"',   # “
            "”": '"',   # ”
            "„": '"',   # „
            "«": '"',   # «
            "»": '"',   # »
            "‘": "'",   # ‘
            "’": "'",   # ’
            "…": "...", # …
            " ": " ",   # NBSP
        }
        text = text.translate(str.maketrans(TYPO_MAP))

        # 0b. Strip stand-alone dashes (em / en / minus / hyphen) that Whisper
        # inserts for spoken pauses. Preserve intra-word hyphens like
        # "kit-kat" or "self-driving" — only dashes with whitespace (or
        # string boundary) on at least one side are removed.
        text = re.sub(r'\s+[—–−\-]+\s+', ' ', text)
        text = re.sub(r'^[—–−\-]+\s+',   '',  text)
        text = re.sub(r'\s+[—–−\-]+$',   '',  text)

        # 1. Verbal commands
        if self._cfg.get("verbal_commands"):
            for pattern, replacement in _VERBAL_RE:
                text = pattern.sub(replacement, text)
            # If the entire result is a delete sentinel, bubble it up
            if text.strip() == "\x00DEL":
                return "\x00DEL"
            text = text.replace("\x00DEL", "")   # embedded sentinels → remove

        # 2. Filler words
        if self._cfg.get("remove_fillers"):
            for w in FILLER_WORDS:
                text = text.replace(w, " ")

        # 3. Normalise whitespace (preserve intentional newlines)
        lines = [" ".join(ln.split()) for ln in text.split("\n")]
        text  = "\n".join(lines)

        # 4. Capitalise after sentence-ending punctuation
        text = re.sub(
            r'(?<=[.!?]\s)([a-záàâãéèêíïóôõöúçñü])',
            lambda m: m.group(1).upper(),
            text,
        )

        return text.strip()

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def _add_to_history(self, text: str):
        self._history.insert(0, text)
        if len(self._history) > MAX_HISTORY:
            self._history.pop()

        preview = text[:52].replace("\n", "↵")
        if len(text) > 52:
            preview += "…"
        self._set_menu_title(self._last_item, f"Last paste — “{preview}”")

    # ══════════════════════════════════════════════════════════════════════════
    # LOCAL LLM DOWNLOAD / PRE-WARM
    # ══════════════════════════════════════════════════════════════════════════

    def _prewarm_local_llm(self):
        """
        Background-warm the local LLM at startup. If it's already cached, this
        is essentially a no-op. If not, we kick off the download with a
        user-visible HUD + system notification, watch the cache directory
        size, and update the HUD label with a percentage as bytes arrive.
        """
        # Skip if AI cleanup is disabled.
        if (self._cfg.get("ai_backend") or "none") == "none":
            return
        with self._llm_dl_lock:
            if self._llm_dl_in_flight:
                return
            self._llm_dl_in_flight = True
        try:
            llm_id = self._ai.llm_id
            repo, expected, label = _local_llm_meta(llm_id)
            already = _is_llm_downloaded(llm_id)

            if not already:
                size_gb = expected / 1_000_000_000
                rumps.notification(
                    "Flow",
                    f"Downloading {label.split('  ·  ')[0]}",
                    f"~{size_gb:.1f} GB — first run only.",
                )
                if self._cfg.get("show_hud", True):
                    self._hud.show("downloading", label_override="Downloading model… 0%")

                # Spawn a watcher that updates the HUD with % progress while
                # the actual model download (started below) writes to disk.
                stop_evt = threading.Event()
                def _watch():
                    while not stop_evt.wait(timeout=1.0):
                        b = _hf_cache_bytes(repo)
                        pct = min(99, int(100 * b / max(1, expected)))
                        self._hud.update_label(f"Downloading model… {pct}%")
                threading.Thread(target=_watch, daemon=True).start()

            # Trigger the actual load (downloads if missing). Reuses the
            # AICleanup lock so a real cleanup request later doesn't double-
            # load the model.
            with self._ai._mlx_lock:
                if (self._ai._mlx_model is None
                        or self._ai._mlx_loaded_repo != repo):
                    try:
                        from mlx_lm import load
                        self._ai._mlx_model, self._ai._mlx_tok = load(repo)
                        self._ai._mlx_loaded_repo = repo
                    except Exception as e:
                        print(f"[prewarm] failed: {e}", file=sys.stderr, flush=True)

            if not already:
                stop_evt.set()
                self._hud.hide()
                rumps.notification(
                    "Flow", "Model ready",
                    f"{label.split('  ·  ')[0]} loaded — AI cleanup is active.",
                )
        finally:
            with self._llm_dl_lock:
                self._llm_dl_in_flight = False

    # ══════════════════════════════════════════════════════════════════════════
    # TELEMETRY
    # ══════════════════════════════════════════════════════════════════════════

    def _emit_timing(self, t: dict):
        """
        Pretty-print a one-line timing summary AND append a CSV row to
        ~/.flow/timings.csv so trends across sessions can be reviewed later.
        """
        # Console line — easy to scan in tail -f
        ai_tag = (
            f" cleanup={t['cleanup_s']:.2f}s"
            if t.get("ai_used") else ""
        )
        kept = "✓" if t.get("ai_kept") else ("∅" if t.get("ai_used") else "-")
        ratio_pct = (
            (t["transcribe_s"] / t["audio_s"]) * 100
            if t["audio_s"] > 0 else 0.0
        )
        print(
            f"[timing] audio={t['audio_s']:5.2f}s "
            f"transcribe={t['transcribe_s']:.2f}s"
            f"{ai_tag} "
            f"paste={t['paste_s']:.2f}s "
            f"total={t['total_s']:.2f}s "
            f"rtf={ratio_pct:.0f}%  "
            f"in→out={t['chars_in']}→{t['chars_out']} "
            f"lang={t['language']} ai={kept}",
            flush=True,
        )

        # CSV append (create with header on first run)
        try:
            CONFIG_DIR.mkdir(exist_ok=True)
            new_file = not TIMINGS_FILE.exists()
            with open(TIMINGS_FILE, "a", encoding="utf-8") as f:
                if new_file:
                    f.write(
                        "timestamp,audio_s,transcribe_s,cleanup_s,paste_s,total_s,"
                        "chars_in,chars_out,language,backend,model,ai_backend,"
                        "ai_used,ai_kept\n"
                    )
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.write(
                    f"{ts},"
                    f"{t['audio_s']:.3f},"
                    f"{t['transcribe_s']:.3f},"
                    f"{t['cleanup_s']:.3f},"
                    f"{t['paste_s']:.3f},"
                    f"{t['total_s']:.3f},"
                    f"{t['chars_in']},"
                    f"{t['chars_out']},"
                    f"{t['language']},"
                    f"{self._backend},"
                    f"{self._model_name},"
                    f"{self._cfg.get('ai_backend','none')},"
                    f"{int(bool(t.get('ai_used')))},"
                    f"{int(bool(t.get('ai_kept')))}\n"
                )
        except Exception as e:
            print(f"[timing] csv write failed: {e}", file=sys.stderr, flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # I/O HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _paste(text: str):
        """
        Paste text at the cursor of the frontmost app:
          save clipboard → write text → Cmd+V (via CGEvent) → restore clipboard
        Uses pbcopy/pbpaste (subprocess, thread-safe) instead of NSPasteboard
        because AppKit APIs crash when called off the main thread on recent
        macOS releases ("Must only be used from the main thread").
        """
        # Save current clipboard
        try:
            old = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=1.0
            ).stdout
        except Exception:
            old = ""

        # Write our text
        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"), timeout=2.0)
        except Exception as e:
            print(f"[paste] pbcopy failed: {e}", file=sys.stderr, flush=True)
            return

        time.sleep(0.06)

        # Cmd+V via Quartz (thread-safe)
        src  = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        v_dn = Quartz.CGEventCreateKeyboardEvent(src, 9, True)   # 9 = 'v'
        v_up = Quartz.CGEventCreateKeyboardEvent(src, 9, False)
        cmd  = Quartz.kCGEventFlagMaskCommand
        Quartz.CGEventSetFlags(v_dn, cmd)
        Quartz.CGEventSetFlags(v_up, cmd)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, v_dn)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, v_up)

        time.sleep(0.18)

        # Restore previous clipboard
        if old:
            try:
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(old.encode("utf-8"), timeout=2.0)
            except Exception:
                pass

    @staticmethod
    def _undo():
        """Simulate Cmd+Z to undo the last paste."""
        src  = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        z_dn = Quartz.CGEventCreateKeyboardEvent(src, 6, True)   # 6 = 'z'
        z_up = Quartz.CGEventCreateKeyboardEvent(src, 6, False)
        cmd  = Quartz.kCGEventFlagMaskCommand
        Quartz.CGEventSetFlags(z_dn, cmd)
        Quartz.CGEventSetFlags(z_up, cmd)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, z_dn)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, z_up)

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _app_path() -> str:
        exe = sys.executable
        if ".app/Contents/" in exe:
            return exe.split(".app/Contents/")[0] + ".app"
        return f"/Applications/{APP_NAME}.app"

    @staticmethod
    def _frontmost_app_bundle() -> str | None:
        """
        Bundle identifier of the frontmost app (used by AICleanup to pick a
        tone automatically: Slack → casual, Mail → formal, IDE → code, etc.).
        Returns None if NSWorkspace can't tell us.
        """
        try:
            ws  = AppKit.NSWorkspace.sharedWorkspace()
            app = ws.frontmostApplication()
            if app is not None:
                bid = app.bundleIdentifier()
                return str(bid) if bid else None
        except Exception:
            pass
        return None

    @staticmethod
    def _check_login_item() -> bool:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every login item'],
            capture_output=True, text=True,
        )
        return "Flow" in r.stdout


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

import fcntl

def _acquire_single_instance_lock():
    """
    Hold an exclusive flock on ~/.flow/.lock for the lifetime of this process.
    Prevents two Flow instances from fighting over the Fn key / mic / paste.
    Returns the file handle (must stay alive) or None if another instance holds it.
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    lock_path = CONFIG_DIR / ".lock"
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.write(str(os.getpid()))
    f.flush()
    return f


if __name__ == "__main__":
    _instance_lock = _acquire_single_instance_lock()
    if _instance_lock is None:
        # Another Flow is already running. Tell the user via a system alert and
        # exit. We can't use rumps here because the app loop hasn't started.
        try:
            subprocess.Popen([
                "osascript", "-e",
                'display alert "Flow is already running" message "Quit the existing Flow from the menu bar before launching another instance."'
            ])
        except Exception:
            pass
        sys.exit(0)

    # Top-level crash logger: any unhandled exception in main thread ends up here
    try:
        FlowApp().run()
    except Exception as _e:
        import traceback
        print(f"[fatal] {_e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        sys.exit(1)
