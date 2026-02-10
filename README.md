# VoicePing iOS Offline Transcribe

[![iOS](https://img.shields.io/badge/iOS-17%2B-black)](#requirements)
[![Swift](https://img.shields.io/badge/Swift-5.9-orange)](#tech-stack)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#license)

A multi-engine iOS speech-to-text app that runs fully on-device.
No cloud transcription, no API keys, and no network dependency after model download.

This project combines WhisperKit, sherpa-onnx, and FluidAudio into one production-oriented app with model switching, live streaming, telemetry, and E2E coverage.

## Why This Project

- Fully offline transcription pipeline from microphone capture to export
- Multiple ASR engines in one app with side-by-side practical benchmarks
- Real-time streaming path (Zipformer) plus high-accuracy offline paths (Whisper, Parakeet)
- Built-in operational safeguards: VAD gating, storage checks, interruption handling, resource telemetry
- Designed for real-device validation, not simulator-only demos

## Benchmark Snapshot (Real iPad, 30s Fixture)

Baseline: `artifacts/e2e/ios/final-verify-20260209-151128`

**11/11 tested, 9 PASS, 2 FAIL, 0 TIMEOUT**

| Model | tok/s | Time | Verdict | Notes |
|---|---:|---:|---|---|
| Moonshine Tiny | 76.5 | 0.8s | PASS | Fastest overall |
| Zipformer Streaming | 75.2 | 0.7s | PASS | Best real-time responsiveness |
| Moonshine Base | 40.6 | 1.4s | PASS | Strong speed/quality balance |
| SenseVoice Small | 16.2 | 3.6s | PASS | Best multilingual punctuation balance |
| Whisper Tiny | 10.6 | 5.5s | PASS | Lightweight Whisper option |
| Parakeet TDT 0.6B | 9.0 | 6.4s | PASS | High-quality punctuation and casing |
| Whisper Base | 7.9 | 26.3s | FAIL | Quality regression (repetitive/hallucinated text) |
| Whisper Small | 0.9 | 65.3s | PASS | Best stable Whisper quality |
| Whisper Large V3 Turbo | 0.24 | 241.8s | PASS | High quality, high latency |
| Whisper Large V3 Turbo (Compressed) | 0.13 | 428.8s | PASS | Slowest passing model |
| Omnilingual 300M | 0.0 | 97.6s | FAIL | Empty or non-English output on English fixture |

Token speed is computed from transcript token count over `duration_ms` in E2E output and is intended for relative comparison.

## Recommended Models by Goal

- Lowest latency: `moonshine-tiny`, `zipformer-20m`
- Fast with better readability: `moonshine-base`, `sensevoice-small`
- Best Whisper quality that currently passes: `whisper-small`
- Highest quality punctuation/casing with good stability: `parakeet-tdt-v3`

## Feature Highlights

### Transcription

- Live microphone transcription with rolling hypothesis and confirmed text
- Streaming and offline decode paths under a unified engine protocol
- Model download, load, and runtime switching in-app

### Reliability

- VAD and RMS gating to reduce silence hallucinations
- Audio session interruption and route-change handling
- Chunked decoding and adaptive scheduling for real-device stability

### UX and Export

- Timestamp support
- Session audio save (`audio.wav`) plus transcript export (`ZIP`)
- Waveform playback with seek and progress visualization
- CPU, memory, and throughput telemetry while recording

## Supported Models

| Model | Engine | Size | Params | Languages |
|---|---|---:|---:|---|
| Whisper Tiny | WhisperKit (CoreML) | ~80 MB | 39M | 99 languages |
| Whisper Base | WhisperKit (CoreML) | ~150 MB | 74M | 99 languages |
| Whisper Small | WhisperKit (CoreML) | ~500 MB | 244M | 99 languages |
| Whisper Large V3 Turbo | WhisperKit (CoreML) | ~600 MB | 809M | 99 languages |
| Whisper Large V3 Turbo (Compressed) | WhisperKit (CoreML) | ~1 GB | 809M | 99 languages |
| Moonshine Tiny | sherpa-onnx offline | ~125 MB | 27M | English |
| Moonshine Base | sherpa-onnx offline | ~280 MB | 61M | English |
| SenseVoice Small | sherpa-onnx offline | ~240 MB | 234M | zh/en/ja/ko/yue |
| Omnilingual 300M | sherpa-onnx offline | ~365 MB | 300M | 1,600+ languages |
| Zipformer Streaming | sherpa-onnx streaming | ~46 MB | 20M | English |
| Parakeet TDT 0.6B | FluidAudio (CoreML) | ~600 MB | 600M | 25 European languages |

## Architecture

Core runtime:

- `WhisperService`: central orchestrator for model lifecycle, recording, decode loop, and UI-facing state
- `ASREngine` protocol with four implementations:
  - `WhisperKitEngine`
  - `SherpaOnnxOfflineEngine`
  - `SherpaOnnxStreamingEngine`
  - `FluidAudioEngine`
- `AudioRecorder` and utilities for WAV write/export and session persistence

UI stack:

- SwiftUI views + view models (`TranscriptionView`, `ModelSetupView`, waveform playback)
- Live metrics and state-driven controls for recording, playback, and export

## Quick Start

### Requirements

- macOS
- Xcode 15+
- iOS 17+ device or simulator
- `xcodegen` (`brew install xcodegen`)

### Setup

```bash
git clone <repo-url>
cd voiceping-ios-offline-transcribe
xcodegen generate
open VoicePingIOSOfflineTranscribe.xcodeproj
```

### Build from CLI

```bash
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'generic/platform=iOS Simulator' build
```

### Unit Tests

```bash
xcodebuild test -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  -only-testing:OfflineTranscriptionTests
```

### Real-Device E2E (iPad)

```bash
IOS_DEVICE_ID=<your-device-udid> scripts/ios-e2e-test.sh
```

Targeted run example:

```bash
IOS_DEVICE_ID=<your-device-udid> scripts/ios-e2e-test.sh whisper-small parakeet-tdt-v3
```

## Flow Validation (Screenshots + Logs)

Use this when you want auditable proof that the full model-load -> inference -> result flow worked.

### 1) Run E2E with an explicit evidence directory

```bash
IOS_DEVICE_ID=<your-device-udid> \
EVIDENCE_DIR=artifacts/e2e/ios/verify-$(date +%Y%m%d-%H%M%S) \
scripts/ios-e2e-test.sh whisper-tiny whisper-small moonshine-base
```

### 2) Validate screen capture artifacts

Each model folder should contain screenshots such as:

- `01_model_loading.png`
- `02_model_loaded.png`
- `03_inference_result.png`

Quick check:

```bash
find artifacts/e2e/ios/verify-* -maxdepth 2 -name '*.png' | wc -l
```

### 3) Validate machine-readable result output

```bash
find artifacts/e2e/ios/verify-* -maxdepth 2 -name result.json -print
```

Inspect pass/fail + transcript snippet:

```bash
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob("artifacts/e2e/ios/verify-*/**/result.json", recursive=True)):
    r=json.load(open(p))
    print(p, "PASS" if r.get("pass") else "FAIL", str(r.get("duration_ms",0))+"ms", (r.get("transcript","")[:80]))
PY
```

### 4) Validate logs (test runner + app inference log)

- `xcodebuild.log`: full XCTest and runner output
- `inference_log.txt`: app-side inference lifecycle log (real-device mode)

Extract key signal lines:

```bash
rg -n "E2E_RESULT|transcribeTestFile|Result written|ERROR" artifacts/e2e/ios/verify-*/**/xcodebuild.log
rg -n "setupModel|TRANSCRIBE|long-audio|E2E audio stats|ERROR" artifacts/e2e/ios/verify-*/**/inference_log.txt
```

### 5) Optional: UI flow validation (interaction-level screenshots)

```bash
EVIDENCE_DIR=artifacts/ui-flow-tests/ios/verify-$(date +%Y%m%d-%H%M%S) \
scripts/ios-ui-flow-tests.sh
```

## Current Known Issues

- `whisper-base` currently fails quality checks on iPad E2E (repetitive/hallucinated output)
- `omnilingual-300m` currently fails English fixture validation (empty or non-English output)

These are tracked as active inference-quality issues, not app crashes.

## Tech Stack

- Swift 5.9
- SwiftUI + SwiftData
- WhisperKit (CoreML)
- sherpa-onnx (ONNX Runtime, local SPM package)
- FluidAudio (CoreML, Parakeet)
- swift-transformers (HuggingFace model fetch)

## Privacy

- Audio and transcripts are processed on-device
- Network is used only for model download
- No cloud transcription or analytics dependency in the ASR path

## License

Apache License 2.0. See `LICENSE`.

Model weights are downloaded at runtime and keep their own licenses. See `NOTICE`.

## Creator

Created by **Akinori Nakajima** ([atyenoria](https://github.com/atyenoria)).
