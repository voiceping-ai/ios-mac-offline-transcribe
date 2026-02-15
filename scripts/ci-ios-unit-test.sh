#!/usr/bin/env bash

# CI unit test runner for iOS: run the OfflineTranscription unit-test target.
# Requires IOS_DESTINATION environment variable.
# Example: IOS_DESTINATION="id=<device-udid>" ./scripts/ci-ios-unit-test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_FILE="$PROJECT_DIR/VoicePingIOSOfflineTranscribe.xcodeproj"
SCHEME="OfflineTranscription"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$PROJECT_DIR/build/DerivedData}"
TEST_FILTER="${IOS_TEST_FILTER:-OfflineTranscriptionTests}"
DESTINATION="${IOS_DESTINATION:-}"

if [ -z "$DESTINATION" ]; then
    echo "ERROR: IOS_DESTINATION environment variable is required."
    echo "Usage: IOS_DESTINATION=\"id=<device-udid>\" $0"
    exit 1
fi

set -o pipefail

XCODEBUILD_ARGS=(
  test
  -project "$PROJECT_FILE"
  -scheme "$SCHEME"
  -destination "$DESTINATION"
  -configuration Debug
  -derivedDataPath "$DERIVED_DATA_PATH"
  -allowProvisioningUpdates
  CODE_SIGNING_ALLOWED=NO
)

if [ -n "${TEST_FILTER:-}" ]; then
  XCODEBUILD_ARGS+=("-only-testing:$TEST_FILTER")
fi

xcodebuild "${XCODEBUILD_ARGS[@]}"
