# VoicePing iOS + macOS Offline Transcribe

Offline-first iOS and macOS transcription app focused on local speech recognition.
All inference runs on-device after model download.

## Current Scope (Code-Accurate)

- Live transcription with confirmed text plus rolling hypothesis.
- Audio source switching:
  - `Voice` (microphone)
  - `Device Audio` (speaker playback via measurement mode)
  - `System` (ReplayKit Broadcast Upload Extension + shared ring buffer)
- In-app model download/load/switch (15 models, 9 engines, multi-backend per Whisper and Qwen3 models).
- Runtime stats while recording (`CPU`, `RAM`, `tok/s`, elapsed audio).
- Recording controls and test-audio transcription (`test_speech.wav`).
- Session history with audio playback, waveform scrubber, and ZIP export.
- Settings toggles for `Voice Activity Detection` and `timestamps`.
- No cloud ASR dependency in runtime path.

Note: translation APIs exist in the service layer, but this repo build is transcription-focused and uses a no-op translation implementation.

## Supported Models

Defined in `OfflineTranscription/Models/ModelInfo.swift` and `OfflineTranscription/Resources/model-catalog.default.json`. 15 model cards across 8 families, with multiple inference backends per Whisper and Qwen3 model.

### Best Engine per Model (sorted by speed)

Model ID links point to the **original model** from the research team. Runtime distribution links are in the "Model Origins" table below.

| Model ID | Engine | Params | Disk | Languages | tok/s | RTF |
|---|---|---:|---:|---|---:|---:|
| [`parakeet-tdt-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | FluidAudio (CoreML) | 600M | ~600 MB | 25 European | 181.8 | 0.011 |
| [`zipformer-20m`](https://github.com/k2-fsa/icefall) | sherpa-onnx streaming | 20M | ~46 MB | English | 39.7 | 0.046 |
| [`whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | Cactus (whisper.cpp) | 39M | 32 MB | 99 languages | 37.8 | 0.051 |
| [`moonshine-tiny`](https://huggingface.co/usefulsensors/moonshine-tiny) | sherpa-onnx offline | 27M | ~125 MB | English | 37.3 | 0.052 |
| [`moonshine-base`](https://huggingface.co/usefulsensors/moonshine-base) | sherpa-onnx offline | 61M | ~280 MB | English | 31.3 | 0.062 |
| [`whisper-base`](https://huggingface.co/openai/whisper-base) | WhisperKit (CoreML) | 74M | ~150 MB | English | 19.6 | 0.114 |
| [`sensevoice-small`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | sherpa-onnx offline | 234M | ~240 MB | zh/en/ja/ko/yue | 15.6 | 0.124 |
| [`whisper-base`](https://huggingface.co/openai/whisper-base) | Cactus (whisper.cpp) | 74M | 60 MB | 99 languages | 13.8 | 0.140 |
| [`whisper-small`](https://huggingface.co/openai/whisper-small) | WhisperKit (CoreML) | 244M | ~500 MB | 99 languages | 6.3 | 0.339 |
| [`qwen3-asr-0.6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Pure C (ARM NEON) | 600M | ~1.8 GB | 30 languages | 5.6 | 0.345 |
| [`qwen3-asr-0.6b-onnx`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | ONNX Runtime (INT8) | 600M | ~1.6 GB | 30 languages | 5.4 | 0.360 |
| [`qwen3-asr-0.6b-mlx`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | MLX (Metal GPU) | 600M | ~400 MB | 30 languages | - | - |
| [`whisper-small`](https://huggingface.co/openai/whisper-small) | Cactus (whisper.cpp) | 244M | 190 MB | 99 languages | 3.9 | 0.492 |
| [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | Cactus (whisper.cpp) | 809M | 574 MB | 99 languages | 0.8 | 2.486 |
| [`omnilingual-300m`](https://huggingface.co/facebook/mms-1b-all) | sherpa-onnx offline | 300M | ~365 MB | 1,600+ languages | - | - |
| [`apple-speech`](https://developer.apple.com/documentation/speech/sfspeechrecognizer) | SFSpeechRecognizer | System | Built-in | 50+ languages | - | - |

### Model Origins

Original models vs runtime distribution formats used by each engine.

| Model Family | Original Author | License | Runtime Distribution |
|---|---|---|---|
| Whisper | [OpenAI](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT | [WhisperKit CoreML](https://huggingface.co/argmaxinc/whisperkit-coreml) · [whisper.cpp GGML](https://huggingface.co/ggerganov/whisper.cpp) |
| Moonshine | [Useful Sensors](https://huggingface.co/usefulsensors/moonshine-base) | MIT | [sherpa-onnx INT8](https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8) |
| SenseVoice | [Alibaba FunAudioLLM](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | Apache 2.0 | [sherpa-onnx INT8](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17) |
| Zipformer | [k2-fsa / icefall](https://github.com/k2-fsa/icefall) | Apache 2.0 | [sherpa-onnx](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17) |
| Omnilingual | [Facebook MMS](https://huggingface.co/facebook/mms-1b-all) | CC-BY-NC 4.0 | [sherpa-onnx INT8](https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12) |
| Parakeet TDT | [NVIDIA NeMo](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | CC-BY 4.0 | FluidAudio CoreML · [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) |
| Qwen3 ASR | [Alibaba Qwen](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Apache 2.0 | [antirez/qwen-asr](https://github.com/antirez/qwen-asr) (Pure C) · [ONNX INT8](https://huggingface.co/jima/qwen3-asr-0.6b-onnx-int8) · [MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit) |
| Apple Speech | [Apple](https://developer.apple.com/documentation/speech) | System built-in | On-device (no download) |

Download URL sources (code-accurate):
- Cactus (whisper.cpp): `OfflineTranscription/Resources/model-catalog.default.json` contains exact Hugging Face `resolve/main/*` artifact URLs + `sha256` checksums.
- sherpa-onnx + Qwen (Pure C/ONNX): `OfflineTranscription/Services/ModelDownloader.swift` constructs Hugging Face `resolve/main/*` URLs (default org: `csukuangfj`).
- WhisperKit: `WhisperKit.download(... from: "argmaxinc/whisperkit-coreml")`.
- FluidAudio: `AsrModels.downloadAndLoad(version: .v3)` downloads from Hugging Face `FluidInference/*` repos (see FluidAudio docs).

### Whisper Backend Comparison (Cactus vs WhisperKit)

| Model | Cactus (whisper.cpp) | WhisperKit (CoreML) | Cactus Advantage |
|:------|---------------------:|--------------------:|:-----------------|
| Whisper Tiny | **1,533 ms** | 14,590 ms | 9.5x faster, 100% reliable |
| Whisper Base | 4,195 ms | 3,425 ms | Multilingual (not .en only) |
| Whisper Small | 14,751 ms | 10,168 ms | Works on 4 GB devices |
| Whisper Large V3 Turbo | 74,582 ms | FAIL (OOM) | Only option on 4 GB devices |

Cactus uses GGML quantized models (Q5_0/Q5_1/Q8_0) with no CoreML compilation step. WhisperKit uses ANE/GPU acceleration but requires 30-120s first-run CoreML compilation and fails on memory-constrained devices.

Benchmarked on iPad Pro 3rd gen (A12X, 4 GB) with 30s test audio. RTF = Real-Time Factor (lower is faster; <1 = faster than real-time). Full benchmark data in `artifacts/benchmarks/`.

`parakeet-tdt-v3` is filtered at runtime when device capability checks fail.

## Architecture

- Orchestrator: `OfflineTranscription/Services/WhisperService.swift`
- Engines (9 total):
  - `CactusEngine` — Whisper family via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (CPU + Metal GPU, GGML quantized)
  - `WhisperKitEngine` — Whisper family via [WhisperKit](https://github.com/argmaxinc/WhisperKit) CoreML (ANE + GPU)
  - `SherpaOnnxOfflineEngine` — Moonshine, SenseVoice, Omnilingual via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ONNX Runtime
  - `SherpaOnnxStreamingEngine` — Zipformer via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ONNX Runtime (100 ms chunks)
  - `FluidAudioEngine` — Parakeet TDT via [FluidAudio](https://github.com/FluidInference/FluidAudio) CoreML
  - `AppleSpeechEngine` — Built-in [SFSpeechRecognizer](https://developer.apple.com/documentation/speech/sfspeechrecognizer)
  - `QwenASREngine` — [Qwen3 ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [antirez/qwen-asr](https://github.com/antirez/qwen-asr) pure C (ARM NEON, 6 threads)
  - `QwenOnnxEngine` — [Qwen3 ASR 0.6B INT8](https://huggingface.co/jima/qwen3-asr-0.6b-onnx-int8) via ONNX Runtime
  - `MLXEngine` — [Qwen3 ASR 0.6B 4-bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit) via [MLX](https://github.com/ml-explore/mlx-swift) Metal GPU (macOS Apple Silicon only)
- Backend selection: `BackendResolver` with automatic fallback (iOS: cactus → legacy; macOS: mlx → legacy)
- Audio capture: `OfflineTranscription/Services/AudioRecorder.swift`
- System capture:
  - `BroadcastUploadExtension/SampleHandler.swift` (ReplayKit)
  - `OfflineTranscription/Services/SystemAudioSource.swift` (shared ring buffer IPC)
- History/export: SwiftData (`TranscriptionRecord`) + `SessionFileManager` + `ZIPExporter`
- UI: `OfflineTranscription/Views/TranscriptionView.swift`, `OfflineTranscription/Views/ModelSetupView.swift`

## Requirements

- macOS
- Xcode 15+
- iOS 17+
- `xcodegen`

## Setup

```bash
git clone <repo-url>
cd ios-mac-offline-transcribe
scripts/setup-ios-deps.sh
scripts/generate-ios-project.sh
open VoicePingIOSOfflineTranscribe.xcodeproj
```

For local signing overrides:

```bash
cp project.local.yml.example project.local.yml
scripts/generate-ios-project.sh
```

## Build

```bash
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'generic/platform=iOS Simulator' build
```

## macOS App

This repo includes a native macOS target: `OfflineTranscriptionMac`.

### Setup (macOS)

```bash
scripts/setup-ios-deps.sh
scripts/setup-macos-deps.sh
scripts/generate-ios-project.sh
```

### Build + Install (macOS)

`--auto-test` is only compiled in `Debug` builds (used for benchmarking).

```bash
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscriptionMac \
  -configuration Debug \
  -derivedDataPath build/DerivedDataMac \
  -destination 'platform=macOS,name=My Mac,arch=arm64' \
  -allowProvisioningUpdates \
  build

rm -rf /Applications/OfflineTranscriptionMac-local.app
ditto build/DerivedDataMac/Build/Products/Debug/OfflineTranscriptionMac.app \
  /Applications/OfflineTranscriptionMac-local.app
```

### Run (macOS)

```bash
open -n -a /Applications/OfflineTranscriptionMac-local.app

# Auto-test a single model (writes an E2E JSON result file)
open -n -a /Applications/OfflineTranscriptionMac-local.app --args \
  --auto-test --model-id qwen3-asr-0.6b-onnx
```

### Model Caching (macOS)

Models are cached locally and persist across app reinstalls.

- sherpa-onnx + Qwen bundles (App Group container):
  - `~/Library/Group Containers/*group.com.voiceping.transcribe*/SherpaModels`
- WhisperKit (Hugging Face Hub) models:
  - `~/Library/Containers/<bundle-id>/Data/Documents/huggingface/models`

If the UI keeps showing **Downloading model...** on every launch, it's usually because the model directory is missing/corrupt, or you are running an older build that cached into a different location.

### Benchmark (macOS)

- E2E result JSON output:
  - `~/Library/Containers/<bundle-id>/Data/tmp/e2e_result_<model_id>.json`
  - (when not sandboxed, it may also write to `/tmp/e2e_result_<model_id>.json`)

Run a per-model E2E sweep:

```bash
EVAL_WAV_PATH=artifacts/benchmarks/long_en_eval.wav \
  scripts/macos-e2e-test.sh --app /Applications/OfflineTranscriptionMac-local.app
```

Latest benchmark table (macOS, 30s fixture): `artifacts/benchmarks/macos-benchmark-2026-02-14.md`.

## Tests and Automation

```bash
scripts/ci-ios-unit-test.sh
scripts/ios-e2e-test.sh
scripts/ios-ui-flow-tests.sh
```

## Privacy

- Audio is processed locally on device.
- Network is used for model downloads only.
- No cloud transcription service is required for runtime inference.

## License

Apache License 2.0. See `LICENSE`.

<!-- BENCHMARK_RESULTS_START -->
### Inference Token Speed Benchmarks

Measured from E2E `result.json` files using a longer English fixture.

Fixture: `artifacts/benchmarks/long_en_eval.wav` (30.00s, 16kHz mono WAV)

#### Evaluation Method

- Per-model E2E runs with the same English fixture on each platform.
- `duration_sec = duration_ms / 1000` from each model `result.json`.
- `Words` is computed from transcript words: `[A-Za-z0-9']+`.
- `tok/s` uses `tokens_per_second` from `result.json` when present; otherwise `Words / duration_sec`.
- `RTF = duration_sec / audio_duration_sec`.

#### iOS Graph

![iOS tokens/sec](artifacts/benchmarks/ios_tokens_per_second.svg)

#### iOS Results

| Model | Engine | Words | Inference (ms) | Tok/s | RTF | Result |
|---|---|---:|---:|---:|---:|---|
| `zipformer-20m` | sherpa-onnx · Streaming | 55 | 856 | 64.24 | 0.03 | PASS |
| `moonshine-tiny` | sherpa-onnx · ONNX Runtime | 58 | 1094 | 53.00 | 0.04 | PASS |
| `moonshine-base` | sherpa-onnx · ONNX Runtime | 58 | 1599 | 36.28 | 0.05 | PASS |
| `sensevoice-small` | sherpa-onnx offline (ONNX Runtime) | 58 | 4190 | 13.84 | 0.14 | PASS |
| `whisper-tiny` | WhisperKit · CoreML | 58 | 18403 | 3.15 | 0.61 | PASS |
| `apple-speech` | - | 0 | n/a | n/a | n/a | SKIP |
| `omnilingual-300m` | - | 0 | n/a | n/a | n/a | SKIP |
| `parakeet-tdt-v3` | - | 0 | n/a | n/a | n/a | SKIP |
| `qwen3-asr-0.6b` | - | 0 | n/a | n/a | n/a | SKIP |
| `qwen3-asr-0.6b-onnx` | - | 0 | n/a | n/a | n/a | SKIP |
| `whisper-base` | - | 0 | n/a | n/a | n/a | SKIP |
| `whisper-large-v3-turbo` | - | 0 | n/a | n/a | n/a | SKIP |
| `whisper-large-v3-turbo-compressed` | - | 0 | n/a | n/a | n/a | SKIP |
| `whisper-small` | - | 0 | n/a | n/a | n/a | SKIP |

#### macOS Graph

![macOS tokens/sec](artifacts/benchmarks/macos_tokens_per_second.svg)

#### macOS Results

| Model | Engine | Words | Inference (ms) | Tok/s | RTF | Result |
|---|---|---:|---:|---:|---:|---|
| `parakeet-tdt-v3` | FluidAudio · CoreML | 58 | 338 | 171.55 | 0.01 | PASS |
| `moonshine-tiny` | sherpa-onnx · ONNX Runtime | 58 | 629 | 92.22 | 0.02 | PASS |
| `zipformer-20m` | sherpa-onnx · Streaming | 55 | 711 | 77.36 | 0.02 | PASS |
| `moonshine-base` | sherpa-onnx · ONNX Runtime | 58 | 977 | 59.34 | 0.03 | PASS |
| `sensevoice-small` | sherpa-onnx · ONNX Runtime | 58 | 2119 | 27.37 | 0.07 | PASS |
| `whisper-tiny` | WhisperKit · CoreML | 66 | 2674 | 24.69 | 0.09 | PASS |
| `whisper-base` | WhisperKit · CoreML | 67 | 2877 | 23.29 | 0.10 | PASS |
| `apple-speech` | Apple Speech · On-device | 30 | 2287 | 13.12 | 0.08 | PASS |
| `whisper-small` | WhisperKit · CoreML | 64 | 7385 | 8.67 | 0.25 | PASS |
| `qwen3-asr-0.6b-onnx` | QwenASR · ONNX Runtime | 58 | 7299 | 7.95 | 0.24 | PASS |
| `qwen3-asr-0.6b` | QwenASR · Pure C | 58 | 10232 | 5.67 | 0.34 | PASS |
| `whisper-large-v3-turbo` | WhisperKit · CoreML | 58 | 30948 | 1.87 | 1.03 | PASS |
| `whisper-large-v3-turbo-compressed` | WhisperKit · CoreML | 58 | 37643 | 1.54 | 1.25 | PASS |
| `omnilingual-300m` | sherpa-onnx · ONNX Runtime | 0 | 78809 | 0.03 | 2.63 | FAIL |

#### Reproduce

1. `rm -rf artifacts/e2e/ios/* artifacts/e2e/macos/*`
2. `TARGET_SECONDS=30 scripts/prepare-long-eval-audio.sh`
3. `EVAL_WAV_PATH=artifacts/benchmarks/long_en_eval.wav scripts/ios-e2e-test.sh`
4. (Optional) `EVAL_WAV_PATH=artifacts/benchmarks/long_en_eval.wav scripts/macos-e2e-test.sh`
5. `python3 scripts/generate-inference-report.py --audio artifacts/benchmarks/long_en_eval.wav --update-readme`

One-command runner: `TARGET_SECONDS=30 scripts/run-inference-benchmarks.sh`

<!-- BENCHMARK_RESULTS_END -->
