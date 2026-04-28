#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Flow — build script
#  Produces:  Flow.app  +  Flow.dmg  (ready to share with colleagues)
#
#  Usage:
#    chmod +x build_dmg.sh
#    bash build_dmg.sh
#
#  Requirements: macOS, Python 3 (comes with Xcode Command Line Tools)
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── Settings ─────────────────────────────────────────────────────────────────
APP_NAME="Flow"
BUNDLE_ID="com.shaungori.flow"
VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$SCRIPT_DIR/build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
DMG_PATH="$SCRIPT_DIR/$APP_NAME.dmg"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Building  $APP_NAME.app  →  $APP_NAME.dmg"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Clean previous build ──────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# ── Copy main script ──────────────────────────────────────────────────────────
cp "$SCRIPT_DIR/flow.py" "$APP_BUNDLE/Contents/Resources/flow.py"
echo "✓  Copied flow.py"

# ── Generate app icon (microphone emoji rendered as PNG → icns) ───────────────
echo "   Generating icon…"
python3 - << 'PYEOF'
import subprocess, os, tempfile, sys

# Use sips (built-in macOS) to make a simple icon.
# We'll render a coloured square with an 'F' as a placeholder.
try:
    from PIL import Image, ImageDraw, ImageFont
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    iconset = "/tmp/Flow.iconset"
    os.makedirs(iconset, exist_ok=True)
    for sz in sizes:
        img = Image.new("RGBA", (sz, sz), (30, 30, 30, 255))
        d = ImageDraw.Draw(img)
        # Draw a simple mic shape
        cx, cy, r = sz//2, sz//2, int(sz*0.22)
        d.ellipse([cx-r, cy-sz//3, cx+r, cy-sz//3+2*r], fill=(220,60,60,255))
        d.rectangle([cx-sz//12, cy-sz//3+r, cx+sz//12, cy+sz//8], fill=(220,60,60,255))
        d.arc([cx-sz//4, cy+sz//8-sz//8, cx+sz//4, cy+sz//4], 180, 0, fill=(220,60,60,255), width=max(1,sz//20))
        d.rectangle([cx-sz//40, cy+sz//4, cx+sz//40, cy+sz//3], fill=(220,60,60,255))
        d.rectangle([cx-sz//6, cy+sz//3, cx+sz//6, cy+sz//3+max(2,sz//40)], fill=(220,60,60,255))
        img.save(f"{iconset}/icon_{sz}x{sz}.png")
        if sz <= 512:
            img2 = img.resize((sz*2,sz*2), Image.LANCZOS)
            img2.save(f"{iconset}/icon_{sz}x{sz}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", "/tmp/Flow.icns"], check=True)
    print("   ✓  Icon created with PIL")
except Exception:
    # Fallback: create a minimal valid icns
    try:
        iconset = "/tmp/Flow_fb.iconset"
        os.makedirs(iconset, exist_ok=True)
        subprocess.run(["sips", "-z", "1024", "1024",
                        "/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns",
                        "--out", f"{iconset}/icon_512x512@2x.png"], capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", "/tmp/Flow.icns"], capture_output=True)
        print("   ✓  Fallback icon used")
    except Exception as e:
        print(f"   ℹ  No icon: {e}")
PYEOF

if [ -f "/tmp/Flow.icns" ]; then
    cp "/tmp/Flow.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
    echo "✓  Icon ready"
fi

# ── Info.plist ────────────────────────────────────────────────────────────────
cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>       <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>        <string>$BUNDLE_ID</string>
  <key>CFBundleVersion</key>           <string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key>        <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleIconFile</key>          <string>AppIcon</string>
  <key>LSUIElement</key>               <true/>
  <key>LSMinimumSystemVersion</key>    <string>13.0</string>
  <key>NSMicrophoneUsageDescription</key>
    <string>Flow needs microphone access to transcribe your voice into text.</string>
  <key>NSAppleEventsUsageDescription</key>
    <string>Flow uses System Events to paste transcribed text into any app.</string>
  <key>NSHumanReadableCopyright</key>
    <string>© 2025 Shaun Gioele · ko-fi.com/shaungori</string>
  <key>NSSupportsSuddenTermination</key><true/>
</dict>
</plist>
PLIST
echo "✓  Info.plist written"

# ── Install helper script (in Resources/) ────────────────────────────────────
# Copied from the standalone install.sh next to this build script. Keeping it
# as a separate file (instead of inline heredoc) means edits to install.sh
# take effect on the next build with no copy/paste.

if [ ! -f "$SCRIPT_DIR/install.sh" ]; then
    echo "✗  $SCRIPT_DIR/install.sh missing — cannot build."
    exit 1
fi
cp "$SCRIPT_DIR/install.sh" "$APP_BUNDLE/Contents/Resources/install.sh"
chmod +x "$APP_BUNDLE/Contents/Resources/install.sh"
echo "✓  install.sh copied (Resources/)"

# ── Compile the C launcher (Flow binary) ─────────────────────────────────────
# This is the actual main bundle executable. Because IT is the process that
# runs Python via dlopen + Py_Main, macOS / TCC identifies the process as
# "Flow" rather than "python3.12".

if ! command -v clang >/dev/null 2>&1; then
    echo "✗  clang not found. Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

clang -O2 -Wall -o "$APP_BUNDLE/Contents/MacOS/$APP_NAME" \
    "$SCRIPT_DIR/flow_launcher.c" \
    -framework CoreFoundation
echo "✓  Compiled C launcher → Contents/MacOS/$APP_NAME"

chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# ── Verify .app structure ─────────────────────────────────────────────────────
echo ""
echo "App bundle contents:"
find "$APP_BUNDLE" -type f | sed "s|$BUILD_DIR/||g" | sort
echo ""

# ── Create DMG ────────────────────────────────────────────────────────────────
echo "Creating DMG…"

# Remove existing DMG
rm -f "$DMG_PATH"

# Use create-dmg if available (nicer result), otherwise fall back to hdiutil
if command -v create-dmg &>/dev/null; then
    echo "   Using create-dmg (Homebrew)…"
    create-dmg \
        --volname   "$APP_NAME" \
        --window-size 540 380 \
        --icon-size  128 \
        --icon "$APP_NAME.app" 140 190 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 400 190 \
        "$DMG_PATH" \
        "$BUILD_DIR/"
else
    echo "   Using hdiutil (built-in)…"
    # Create a staging folder where ONLY the installer + README are visible.
    # Flow.app is hidden inside .payload/ so users can't bypass the installer
    # and hit the "damaged file" Gatekeeper error.
    STAGING="$BUILD_DIR/dmg_staging"
    mkdir -p "$STAGING/.payload"
    cp -R "$APP_BUNDLE" "$STAGING/.payload/"

    # Pre-built assets that the Install Flow.command will extract:
    # • python.tar.gz       — python-build-standalone 3.12.13 (~17 MB)
    # • venv-libs.tar.gz    — pre-installed Python deps (~343 MB)
    # If these are missing the installer falls back to online mode.
    if [ -f "$SCRIPT_DIR/build_assets/python.tar.gz" ]; then
        cp "$SCRIPT_DIR/build_assets/python.tar.gz" "$STAGING/.payload/python.tar.gz"
        echo "✓  Bundled python.tar.gz   ($(du -h "$STAGING/.payload/python.tar.gz" | cut -f1))"
    fi
    if [ -f "$SCRIPT_DIR/build_assets/venv-libs.tar.gz" ]; then
        cp "$SCRIPT_DIR/build_assets/venv-libs.tar.gz" "$STAGING/.payload/venv-libs.tar.gz"
        echo "✓  Bundled venv-libs.tar.gz ($(du -h "$STAGING/.payload/venv-libs.tar.gz" | cut -f1))"
    fi

    # ── Install helper: copies Flow to /Applications, removes quarantine,
    #    re-signs ad-hoc, and launches.
    cat > "$STAGING/Install Flow.command" << 'INSTALLCMD'
#!/bin/bash
# Flow installer — verbose, with quarantine & re-sign handling.
# Logs everything to ~/.flow/installer.log for post-mortem.
set -u

DMG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_APP="$DMG_DIR/.payload/Flow.app"
DEST_APP="/Applications/Flow.app"
LOG="$HOME/.flow/installer.log"

mkdir -p "$HOME/.flow"
exec > >(tee -a "$LOG") 2>&1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Flow installer  ·  $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DMG dir:   $DMG_DIR"
echo "SRC app:   $SRC_APP"
echo "DEST app:  $DEST_APP"
echo ""

# ── 1. Sanity: payload present? ─────────────────────────────────────────────
if [ ! -d "$SRC_APP" ]; then
    echo "✗ Flow.app not found at $SRC_APP"
    osascript -e 'display alert "Flow installer error" message "Flow.app payload missing from DMG. Re-download the DMG and try again." as critical'
    read -p "Press Enter to close… " _
    exit 1
fi
echo "✓ Payload found ($(du -sh "$SRC_APP" 2>/dev/null | cut -f1))"

# ── 2. Replace any existing /Applications/Flow.app ──────────────────────────
if [ -d "$DEST_APP" ]; then
    echo "→ Existing Flow.app in /Applications — replacing…"
    if ! rm -rf "$DEST_APP" 2>/dev/null; then
        # Maybe owned by another user / locked. Try with admin.
        echo "  (rm failed without admin; asking for password)"
        osascript -e "do shell script \"rm -rf '$DEST_APP'\" with administrator privileges" || {
            osascript -e 'display alert "Could not remove old Flow" message "The existing /Applications/Flow.app could not be deleted. Quit Flow if running, then retry." as critical'
            exit 2
        }
    fi
fi

# ── 3. Copy ─────────────────────────────────────────────────────────────────
echo "→ Copying Flow.app to /Applications…"
if ! cp -R "$SRC_APP" "$DEST_APP" 2>&1; then
    echo "✗ cp failed."
    osascript -e 'display alert "Install failed" message "Could not copy Flow.app to /Applications. Check ~/.flow/installer.log." as critical'
    read -p "Press Enter to close… " _
    exit 3
fi

if [ ! -x "$DEST_APP/Contents/MacOS/Flow" ]; then
    echo "✗ Copy looks incomplete — main binary missing."
    osascript -e 'display alert "Install incomplete" message "Flow.app copy is incomplete. See ~/.flow/installer.log." as critical'
    exit 4
fi
echo "✓ Copy complete ($(du -sh "$DEST_APP" 2>/dev/null | cut -f1))"

# ── 4. Strip quarantine + re-sign ad-hoc ────────────────────────────────────
echo "→ Removing com.apple.quarantine recursively…"
xattr -dr com.apple.quarantine "$DEST_APP" 2>/dev/null || true

echo "→ Re-applying ad-hoc signature…"
if ! codesign --force --deep --sign - "$DEST_APP" 2>&1; then
    echo "  (codesign failed; ignoring — bundle may still launch)"
fi

# ── 5. Bundled Python + venv extraction (offline install) ───────────────────
PORTABLE_PY="$HOME/.flow/python"
VENV_DIR="$HOME/.flow/venv"
LOCK_FILE="$HOME/.flow/.install_done"

if [ -f "$DMG_DIR/.payload/python.tar.gz" ] && [ -f "$DMG_DIR/.payload/venv-libs.tar.gz" ]; then
    echo ""
    echo "→ Installing bundled Python runtime…"
    rm -rf "$PORTABLE_PY" "$PORTABLE_PY.tmp"
    mkdir -p "$PORTABLE_PY.tmp"
    if ! tar -xzf "$DMG_DIR/.payload/python.tar.gz" -C "$PORTABLE_PY.tmp"; then
        osascript -e 'display alert "Install failed" message "Could not extract bundled Python." as critical'
        exit 5
    fi
    mv "$PORTABLE_PY.tmp/python" "$PORTABLE_PY"
    rm -rf "$PORTABLE_PY.tmp"

    # ── Critical: macOS Gatekeeper blocks the freshly-extracted Python and
    # all bundled .dylib files because they inherit com.apple.quarantine from
    # the DMG. Strip recursively, then re-apply a fresh ad-hoc signature.
    echo "→ De-quarantining Python runtime…"
    xattr -dr com.apple.quarantine "$PORTABLE_PY" 2>/dev/null || true
    echo "→ Ad-hoc signing Python binaries (this can take ~30 s)…"
    codesign --force --deep --sign - "$PORTABLE_PY/bin/python3.12" 2>/dev/null || true
    # Sign every dylib too (Gatekeeper checks each on dlopen)
    find "$PORTABLE_PY" -name "*.dylib" -o -name "*.so" 2>/dev/null | while read -r f; do
        codesign --force --sign - "$f" 2>/dev/null
    done

    echo "  ✓ Python at $PORTABLE_PY"
    if ! "$PORTABLE_PY/bin/python3" --version 2>&1; then
        osascript -e 'display alert "Python blocked by macOS" message "The bundled Python could not run. Open Terminal and execute:\n\nxattr -dr com.apple.quarantine ~/.flow/python\ncodesign --force --deep --sign - ~/.flow/python\n\nthen re-run the installer." as critical'
        exit 7
    fi

    echo "→ Creating venv shell…"
    rm -rf "$VENV_DIR"
    "$PORTABLE_PY/bin/python3" -m venv "$VENV_DIR" --without-pip
    rm -rf "$VENV_DIR/lib"     # we'll replace it with the bundled site-packages

    echo "→ Installing bundled Python libraries (~343 MB)…"
    if ! tar -xzf "$DMG_DIR/.payload/venv-libs.tar.gz" -C "$VENV_DIR"; then
        osascript -e 'display alert "Install failed" message "Could not extract bundled libraries." as critical'
        exit 6
    fi

    echo "→ De-quarantining libraries (.dylib / .so)…"
    xattr -dr com.apple.quarantine "$VENV_DIR" 2>/dev/null || true
    echo "→ Ad-hoc signing native extensions (this takes 1–2 min)…"
    find "$VENV_DIR/lib" \( -name "*.dylib" -o -name "*.so" \) 2>/dev/null | while read -r f; do
        codesign --force --sign - "$f" 2>/dev/null
    done
    echo "  ✓ Libraries at $VENV_DIR/lib"

    echo "→ Sanity check imports…"
    if "$VENV_DIR/bin/python3" -c "import mlx_whisper, mlx_lm, rumps, sounddevice, numpy, AppKit, Quartz; print('  imports OK')" 2>&1; then
        touch "$LOCK_FILE"
        echo "  ✓ Setup complete."
    else
        echo "  ⚠ One or more modules failed to import. Check the log above."
        osascript -e 'display alert "Some modules failed to load" message "See ~/.flow/installer.log for details. Flow may still work but AI cleanup might be unavailable." as warning'
    fi
else
    echo ""
    echo "→ No bundled assets found in DMG — Flow will install dependencies on"
    echo "  first launch (online mode)."
fi

# ── 6. Launch ───────────────────────────────────────────────────────────────
echo ""
echo "✅ Flow installed in /Applications."
echo ""
echo "Next:"
echo "  • macOS will prompt for Accessibility — grant it."
echo "  • Hold Fn to dictate, release to transcribe."
echo "  • The first dictation downloads the Whisper model (~1.5 GB)."
echo ""
echo "Launching Flow…"
open "$DEST_APP"
sleep 2
echo ""
echo "Log: $LOG"
read -p "Press Enter to close this window… " _
INSTALLCMD
    chmod +x "$STAGING/Install Flow.command"

    # ── Plain-text README for the impatient ─────────────────────────────────
    cat > "$STAGING/READ ME FIRST.txt" << 'README'
Flow — local voice dictation for macOS
══════════════════════════════════════

EASIEST INSTALL
───────────────
Double-click  "Install Flow.command"
It copies Flow to /Applications, removes the macOS quarantine flag, and
launches the app.

(If macOS warns that the .command can't be opened, right-click → Open →
confirm. You only need to do this once.)


IF THE INSTALLER FAILS  (Gatekeeper crash)
─────────────────────────────────────────
Open Terminal and run:

    xattr -dr com.apple.quarantine ~/.flow ~/Downloads/Flow.dmg
    sudo xattr -dr com.apple.quarantine /Applications/Flow.app

Then double-click "Install Flow.command" again.


REQUIREMENTS
────────────
• Apple Silicon Mac (M1 / M2 / M3 / M4)
• macOS 13 (Ventura) or newer — recommended 14+
• Python 3.10+ via Homebrew:    brew install python@3.12

The first launch downloads Python dependencies into ~/.flow/venv (~5 min,
one-time). The first dictation downloads the Whisper model (~1.5 GB) and
optionally an LLM for cleanup (~1.8 GB) — both shown in a HUD with a
progress bar.


USAGE
─────
Hold Fn          push-to-talk (release to transcribe and paste)
Press Fn twice   hands-free mode (auto-segments by silence)

Menu bar → Language, Whisper Model, Backend, AI Cleanup, Cleanup Tone,
Local LLM, Edit Dictionary…
README

    hdiutil create \
        -volname   "$APP_NAME" \
        -srcfolder "$STAGING" \
        -ov \
        -format    UDZO \
        "$DMG_PATH"

    rm -rf "$STAGING"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Done!"
echo ""
echo "  DMG:   $DMG_PATH"
echo ""
echo "  To install:"
echo "    1. Open Flow.dmg"
echo "    2. Drag Flow → Applications"
echo "    3. Launch Flow"
echo "    4. Grant Accessibility access when prompted:"
echo "       System Settings → Privacy & Security → Accessibility"
echo ""
echo "  First launch installs Python deps automatically (~5 min)."
echo "  The Whisper model (~3 GB for large-v3) downloads on first dictation."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
