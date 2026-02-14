#!/usr/bin/env bash

# Adds macOS slices to the SherpaOnnxKit binary XCFramework dependencies.
# Must be run AFTER setup-ios-deps.sh (which downloads the iOS xcframeworks).
#
# Downloads:
# 1. macOS static xcframework archive (sherpa-onnx only, ~6.5 MB)
# 2. Matching onnxruntime static lib (universal2, ~32 MB)
#
# IMPORTANT: The macOS sherpa-onnx static build requires ORT API v23 (ORT 1.23.x),
# but the iOS archive bundles ORT 1.17.x (API v17). We must download a matching
# onnxruntime for the macOS slice to avoid an API version mismatch crash.

set -euo pipefail

SHERPA_VERSION="${SHERPA_VERSION:-1.12.23}"
ORT_VERSION="${ORT_VERSION:-1.23.2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$REPO_DIR/LocalPackages/SherpaOnnxKit"

SHERPA_XC="$PKG_DIR/sherpa-onnx.xcframework"
ONNX_XC="$PKG_DIR/onnxruntime.xcframework"

# Verify iOS deps were run first
if [ ! -d "$SHERPA_XC/ios-arm64" ]; then
  echo "ERROR: iOS xcframeworks not found. Run setup-ios-deps.sh first." >&2
  exit 1
fi

if [ ! -d "$ONNX_XC/macos-arm64_x86_64" ]; then
  echo "ERROR: onnxruntime.xcframework macOS slice not found. Re-run setup-ios-deps.sh." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# --- Download sherpa-onnx macOS static xcframework ---
ARCHIVE="$WORK_DIR/sherpa-onnx-macos-static.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/sherpa-onnx-v${SHERPA_VERSION}-macos-xcframework-static.tar.bz2"

echo "Downloading sherpa-onnx macOS static xcframework v${SHERPA_VERSION}..."
curl -L --fail -o "$ARCHIVE" "$URL"
tar -xjf "$ARCHIVE" -C "$WORK_DIR"

SRC_XC="$WORK_DIR/sherpa-onnx-v${SHERPA_VERSION}-macos-xcframework-static/sherpa-onnx.xcframework"
SRC_MACOS="$SRC_XC/macos-arm64_x86_64"

if [ ! -f "$SRC_MACOS/libsherpa-onnx.a" ]; then
  echo "ERROR: libsherpa-onnx.a not found in macOS static archive" >&2
  exit 1
fi

# --- Download matching onnxruntime for macOS ---
ORT_ZIP="$WORK_DIR/ort-macos.zip"
ORT_URL="https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${ORT_VERSION}/onnxruntime-osx-universal2-static_lib-${ORT_VERSION}.zip"

echo "Downloading onnxruntime macOS static lib v${ORT_VERSION}..."
curl -L --fail -o "$ORT_ZIP" "$ORT_URL"
unzip -q "$ORT_ZIP" -d "$WORK_DIR"

ORT_STATIC="$WORK_DIR/onnxruntime-osx-universal2-static_lib-${ORT_VERSION}/lib/libonnxruntime.a"
if [ ! -f "$ORT_STATIC" ]; then
  echo "ERROR: libonnxruntime.a not found in ORT static archive" >&2
  exit 1
fi

# --- Add macOS slice to sherpa-onnx.xcframework ---

MACOS_SLICE="$SHERPA_XC/macos-arm64_x86_64"
rm -rf "$MACOS_SLICE"
mkdir -p "$MACOS_SLICE"

# Copy the static library, create consistent naming with iOS slices
cp "$SRC_MACOS/libsherpa-onnx.a" "$MACOS_SLICE/sherpa-onnx.a"
ln -s sherpa-onnx.a "$MACOS_SLICE/libsherpa-onnx.a"

# Flatten headers and add modulemap (matching iOS slice layout)
mkdir -p "$MACOS_SLICE/Headers"
cp "$SRC_MACOS/Headers/sherpa-onnx/c-api/c-api.h" "$MACOS_SLICE/Headers/c-api.h"
cat > "$MACOS_SLICE/Headers/module.modulemap" <<'EOF'
module sherpa_onnx {
    header "c-api.h"
    export *
}
EOF

# Write complete Info.plist with all 3 slices (idempotent)
cat > "$SHERPA_XC/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>BinaryPath</key>
			<string>sherpa-onnx.a</string>
			<key>HeadersPath</key>
			<string>Headers</string>
			<key>LibraryIdentifier</key>
			<string>ios-arm64_x86_64-simulator</string>
			<key>LibraryPath</key>
			<string>sherpa-onnx.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
		<dict>
			<key>BinaryPath</key>
			<string>sherpa-onnx.a</string>
			<key>HeadersPath</key>
			<string>Headers</string>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>sherpa-onnx.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>BinaryPath</key>
			<string>sherpa-onnx.a</string>
			<key>HeadersPath</key>
			<string>Headers</string>
			<key>LibraryIdentifier</key>
			<string>macos-arm64_x86_64</string>
			<key>LibraryPath</key>
			<string>sherpa-onnx.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>macos</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
EOF

echo "  Added macOS slice to sherpa-onnx.xcframework"

# --- Replace macOS onnxruntime with matching version ---
# The iOS archive bundles ORT 1.17.x, but macOS sherpa-onnx needs ORT 1.23.x.
echo "  Replacing macOS onnxruntime slice with ORT v${ORT_VERSION}..."
cp "$ORT_STATIC" "$ONNX_XC/macos-arm64_x86_64/onnxruntime.a"

# Write complete Info.plist (no HeadersPath — onnxruntime is link-only,
# Headers would cause modulemap collision with sherpa-onnx).
cat > "$ONNX_XC/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>onnxruntime.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64_x86_64-simulator</string>
			<key>LibraryPath</key>
			<string>onnxruntime.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
		<dict>
			<key>LibraryIdentifier</key>
			<string>macos-arm64_x86_64</string>
			<key>LibraryPath</key>
			<string>onnxruntime.a</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>macos</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
EOF

echo "  Replaced macOS onnxruntime with ORT v${ORT_VERSION}"
echo "SherpaOnnxKit macOS dependencies prepared."
