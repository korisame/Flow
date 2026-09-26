#!/bin/bash
# Build a fresh staging bundle only. Never replaces the installed application.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")" && pwd)"
BIN="$PROJ/.build/release/FlowApp"
[ -f "$BIN" ] || { echo "Build release first"; exit 1; }
STAGE="$(mktemp -d "$PROJ/../stage-XXXXXX")"
APP="$STAGE/FlowSwift.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PROJ/Resources/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN" "$APP/Contents/MacOS/FlowApp"
cp "$PROJ/Resources/punctuation_worker.py" "$APP/Contents/Resources/"
cp /Applications/FlowSwift.app/Contents/Resources/AppIcon.icns "$APP/Contents/Resources/"
for RESOURCE in "$PROJ"/.build/release/*.bundle; do
    [ ! -d "$RESOURCE" ] || cp -R "$RESOURCE" "$APP/Contents/Resources/"
done
SIGNING_IDENTITY="${CODESIGN_IDENTITY:--}"
codesign --force --sign "$SIGNING_IDENTITY" "$APP"
codesign --verify --strict "$APP"
echo "$APP"
