#!/usr/bin/env bash

# Creates a macOS DMG installer from the built .app bundle.
# Expects the app to be at the standard Release derived data path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

APP_PATH="${APP_PATH:-$PROJECT_DIR/build/DerivedData/Build/Products/Release/OfflineTranscriptionMac.app}"
VERSION="${GITHUB_REF_NAME:-dev}"
DMG_PATH="$PROJECT_DIR/build/VoicePingOfflineTranscribe-${VERSION}.dmg"
STAGING="$PROJECT_DIR/build/dmg-staging"

if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: App bundle not found at $APP_PATH" >&2
  echo "Build with: xcodebuild build -scheme OfflineTranscriptionMac -configuration Release" >&2
  exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "$APP_PATH" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create \
  -volname "VoicePing Offline Transcribe" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

rm -rf "$STAGING"
echo "DMG created at: $DMG_PATH"
