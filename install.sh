#!/bin/bash
# Flow first-run installer — self-contained.
# If Python 3.10+ isn't already on the system, downloads a portable Python
# from python-build-standalone (no Homebrew, no sudo, no admin rights).
set -u

VENV="$HOME/.flow/venv"
PORTABLE_PY="$HOME/.flow/python"
LOCK="$HOME/.flow/.install_done"
INSTALL_LOG="$HOME/.flow/install.log"

# Pinned python-build-standalone release. Update when needed; the URL is
# stable as long as the tag/version pair exists in the repo.
PB_VER="3.12.13"
PB_TAG="20260414"
PB_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PB_TAG}/cpython-${PB_VER}+${PB_TAG}-aarch64-apple-darwin-install_only.tar.gz"

mkdir -p "$HOME/.flow"

log()   { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$INSTALL_LOG"; }
notify(){ osascript -e "display notification \"$1\" with title \"Flow\"" 2>/dev/null || true; }
alert() { osascript -e "display alert \"$1\" message \"$2\" as critical" 2>/dev/null || true; }

# ── Step 1 — find or fetch Python 3.10+ ─────────────────────────────────────
find_system_python() {
    for cmd in \
        /opt/homebrew/bin/python3.12 \
        /opt/homebrew/bin/python3.11 \
        /opt/homebrew/bin/python3.10 \
        /usr/local/bin/python3.12 \
        /usr/local/bin/python3.11 \
        /usr/local/bin/python3.10 \
        python3.12 python3.11 python3.10; do
        if command -v "$cmd" &>/dev/null; then
            v=$("$cmd" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null)
            [ "$v" = "True" ] && { echo "$cmd"; return; }
        fi
    done
    echo ""
}

ensure_portable_python() {
    if [ -x "$PORTABLE_PY/bin/python3" ]; then
        log "Portable Python already present at $PORTABLE_PY"
        return 0
    fi
    log "No system Python 3.10+ found."
    log "Downloading portable Python ${PB_VER} (~17 MB) from"
    log "  $PB_URL"
    notify "Downloading Python (~17 MB)…"

    rm -rf "$PORTABLE_PY.tmp" "$PORTABLE_PY.tar.gz"
    mkdir -p "$PORTABLE_PY.tmp"

    # Up to 3 retries with exponential back-off. --fail makes curl exit non-zero
    # on HTTP errors (otherwise we'd get a 0-byte file on 404). --connect-timeout
    # caps DNS/handshake time. --retry handles transient TCP/TLS hiccups.
    local ok=0
    for attempt in 1 2 3; do
        log "  → curl attempt $attempt"
        if curl --fail --location --silent --show-error \
                --connect-timeout 15 --max-time 600 \
                --retry 2 --retry-delay 3 \
                -o "$PORTABLE_PY.tar.gz" \
                "$PB_URL" 2>>"$INSTALL_LOG"; then
            ok=1
            break
        fi
        log "  curl failed (attempt $attempt). Retrying…"
        sleep $((attempt * 2))
    done
    if [ $ok -ne 1 ]; then
        log "ERROR: could not download Python."
        log "  Network problem? Check that this Mac has internet access."
        log "  Tarball URL: $PB_URL"
        rm -rf "$PORTABLE_PY.tmp" "$PORTABLE_PY.tar.gz"
        return 1
    fi

    local downloaded=$(stat -f%z "$PORTABLE_PY.tar.gz" 2>/dev/null || echo 0)
    log "Downloaded ${downloaded} bytes."
    if [ "$downloaded" -lt 5000000 ]; then
        log "ERROR: download too small — likely a redirect/HTML error page."
        head -c 500 "$PORTABLE_PY.tar.gz" >> "$INSTALL_LOG" 2>&1
        rm -rf "$PORTABLE_PY.tmp" "$PORTABLE_PY.tar.gz"
        return 1
    fi

    log "Extracting…"
    if ! tar -xzf "$PORTABLE_PY.tar.gz" -C "$PORTABLE_PY.tmp" 2>>"$INSTALL_LOG"; then
        log "ERROR: tar extraction failed."
        rm -rf "$PORTABLE_PY.tmp" "$PORTABLE_PY.tar.gz"
        return 1
    fi
    rm -f "$PORTABLE_PY.tar.gz"

    # The tarball contains a top-level "python/" directory.
    if [ -d "$PORTABLE_PY.tmp/python" ]; then
        rm -rf "$PORTABLE_PY"
        mv "$PORTABLE_PY.tmp/python" "$PORTABLE_PY"
        rm -rf "$PORTABLE_PY.tmp"
        log "Portable Python installed at $PORTABLE_PY"
        # Sanity check
        if "$PORTABLE_PY/bin/python3" -c "import sys; print('  python OK', sys.version.split()[0])" >>"$INSTALL_LOG" 2>&1; then
            return 0
        fi
        log "ERROR: portable python binary won't run."
        return 1
    fi
    log "ERROR: tarball layout unexpected (no python/ at top level)."
    rm -rf "$PORTABLE_PY.tmp"
    return 1
}

PYTHON=$(find_system_python)
if [ -z "$PYTHON" ]; then
    if ensure_portable_python; then
        PYTHON="$PORTABLE_PY/bin/python3"
    else
        alert "Flow install failed" "Could not find or download Python.\nCheck ~/.flow/install.log for details."
        exit 1
    fi
fi
log "Using Python: $PYTHON ($("$PYTHON" --version 2>&1))"

notify "Installing Flow components (~5 min, one-time)…"

# ── Step 2 — venv + dependency install ──────────────────────────────────────
{
    set -e
    log "Creating venv…"
    "$PYTHON" -m venv "$VENV"

    log "Upgrading pip…"
    "$VENV/bin/pip" install --quiet --upgrade pip

    log "Installing Python packages…"
    "$VENV/bin/pip" install --quiet \
        "faster-whisper" \
        "mlx-whisper" \
        "mlx-lm" \
        "silero-vad" \
        "sounddevice" \
        "numpy" \
        "rumps" \
        "pyobjc" \
        "pyobjc-framework-Quartz" \
        "pyobjc-framework-Cocoa"

    log "All packages installed OK"
} >> "$INSTALL_LOG" 2>&1

if grep -q "All packages installed OK" "$INSTALL_LOG" 2>/dev/null; then
    touch "$LOCK"
    notify "Flow ready — first dictation will download Whisper (~1.5 GB)."
    exit 0
else
    alert "Flow install failed" "Open ~/.flow/install.log to see the error, then delete ~/.flow and relaunch Flow."
    exit 1
fi
