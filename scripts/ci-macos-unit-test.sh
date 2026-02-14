#!/usr/bin/env bash

# CI unit test runner for macOS: run the OfflineTranscriptionMac unit-test target.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_FILE="$PROJECT_DIR/VoicePingIOSOfflineTranscribe.xcodeproj"
SCHEME="OfflineTranscriptionMac"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$PROJECT_DIR/build/DerivedData}"
TEST_FILTER="${MACOS_TEST_FILTER:-OfflineTranscriptionMacTests}"

set -o pipefail

XCODEBUILD_ARGS=(
  test
  -project "$PROJECT_FILE"
  -scheme "$SCHEME"
  -configuration Debug
  -derivedDataPath "$DERIVED_DATA_PATH"
  CODE_SIGNING_ALLOWED=NO
)

if [ -n "${TEST_FILTER:-}" ]; then
  XCODEBUILD_ARGS+=("-only-testing:$TEST_FILTER")
fi

xcodebuild "${XCODEBUILD_ARGS[@]}"
