# Offline Transcription (iOS)

iOS app for **fully offline speech-to-text transcription** — all inference runs on-device with no cloud dependency.

Record speech from the microphone and transcribe it locally using multiple ASR engines (Whisper, Moonshine, SenseVoice, Zipformer, Parakeet). Models are downloaded once from HuggingFace, then everything works completely offline.

## Features

- Real-time microphone recording with live transcript rendering
- 4 ASR engine backends with in-app model switching
- On-device model download with progress tracking
- Streaming transcription (Zipformer transducer, endpoint-based)
- Voice Activity Detection (VAD) toggle
- Optional timestamp display
- Session audio saving as WAV (PCM 16kHz mono 16-bit)
- Audio playback with 200-bar waveform scrubber
- ZIP export of session (transcription + audio)
- AVAudioSession interruption + route change handling
- Live audio energy visualization
- CPU / memory / tokens-per-second telemetry display
- Storage guard before large model downloads

## Supported Models (11)

| Model | Engine | Size | Params | Languages |
|-------|--------|------|--------|-----------|
| Whisper Tiny | WhisperKit (CoreML) | ~80 MB | 39M | 99 languages |
| Whisper Base | WhisperKit (CoreML) | ~150 MB | 74M | 99 languages |
| Whisper Small | WhisperKit (CoreML) | ~500 MB | 244M | 99 languages |
| Whisper Large V3 Turbo | WhisperKit (CoreML) | ~600 MB | 809M | 99 languages |
| Whisper Large V3 Turbo (Compressed) | WhisperKit (CoreML) | ~1 GB | 809M | 99 languages |
| Moonshine Tiny | sherpa-onnx | ~125 MB | 27M | English |
| Moonshine Base | sherpa-onnx | ~280 MB | 61M | English |
| SenseVoice Small | sherpa-onnx | ~240 MB | 234M | zh/en/ja/ko/yue |
| Omnilingual 300M | sherpa-onnx | ~365 MB | 300M | 1,600+ languages |
| Zipformer Streaming | sherpa-onnx | ~46 MB | 20M | English |
| Parakeet TDT 0.6B | FluidAudio (CoreML) | ~600 MB | 600M | 25 European languages |

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   OfflineTranscriptionApp                       │
│                      (SwiftUI App)                              │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ TranscriptionView│  │ WaveformPlayback │  │ ModelSetupView│ │
│  │                  │  │ View             │  │              │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘ │
│           │                     │                    │         │
│  ┌────────▼─────────┐  ┌───────▼──────────┐         │         │
│  │ Transcription    │  │ AudioPlayer      │         │         │
│  │ ViewModel        │  │ ViewModel        │         │         │
│  └────────┬─────────┘  └─────────────────-┘         │         │
│           │                                          │         │
├───────────▼──────────────────────────────────────────▼─────────┤
│                      WhisperService                             │
│             (Orchestrator — coordinates all services)           │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ASREngine Protocol                                        │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │  │
│  │  │ WhisperKit   │ │ SherpaOnnx   │ │ SherpaOnnx       │  │  │
│  │  │ Engine       │ │ OfflineEngine│ │ StreamingEngine  │  │  │
│  │  │ (CoreML)     │ │ (Moonshine,  │ │ (Zipformer)      │  │  │
│  │  │              │ │  SenseVoice, │ │                  │  │  │
│  │  │              │ │  Omnilingual)│ │                  │  │  │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘  │  │
│  │  ┌──────────────┐                                         │  │
│  │  │ FluidAudio   │                                         │  │
│  │  │ Engine       │                                         │  │
│  │  │ (Parakeet)   │                                         │  │
│  │  └──────────────┘                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ AudioRecorder  │  │ SessionFile      │  │ SystemMetrics │  │
│  │ (AVAudioEngine)│  │ Manager          │  │ (CPU/Memory)  │  │
│  └────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                │
│  ┌────────────────┐  ┌──────────────────┐                      │
│  │ ModelDownloader│  │ WAVWriter +      │                      │
│  │ (URLSession)   │  │ ZIPExporter      │                      │
│  └────────────────┘  └──────────────────┘                      │
└───────────────────────────────────────────────────────────────┘
```

### ASR Engines

- **WhisperKitEngine** — Runs Whisper models via CoreML on the Apple Neural Engine. Chunked offline transcription with eager text confirmation.
- **SherpaOnnxOfflineEngine** — Runs Moonshine, SenseVoice, and Omnilingual models via ONNX Runtime. Batch inference on accumulated audio buffers.
- **SherpaOnnxStreamingEngine** — Runs Zipformer transducer via ONNX Runtime with 100ms polling and endpoint detection.
- **FluidAudioEngine** — Runs NVIDIA Parakeet-TDT via CoreML using the FluidAudio SDK. Best English WER (2.5%), 25 European languages.

## Setup

**Requirements:** macOS, Xcode 15+, iOS 17+ simulator or device, XcodeGen (`brew install xcodegen`)

```bash
git clone <repo-url>
cd repo-ios-transcription-only

# Generate Xcode project
xcodegen generate
open OfflineTranscription.xcodeproj
```

Build from CLI:
```bash
xcodebuild -project OfflineTranscription.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'generic/platform=iOS Simulator' build
```

## Testing

```bash
# Unit tests (~110 tests, 8 suites)
xcodebuild test -scheme OfflineTranscription \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  -only-testing:OfflineTranscriptionTests
```

## Tech Stack

- Swift 5.9 + SwiftUI + SwiftData
- WhisperKit (CoreML inference, pinned revision)
- sherpa-onnx (local SPM package, ONNX Runtime)
- FluidAudio (CoreML, Parakeet-TDT)
- swift-transformers (HuggingFace Hub)
- iOS 17.0+

## Dependencies

| Package | Purpose |
|---------|---------|
| WhisperKit | Whisper CoreML inference |
| SherpaOnnxKit | sherpa-onnx local SPM package |
| FluidAudio | Parakeet-TDT CoreML inference |
| swift-transformers | HuggingFace Hub model downloads |
| swift-collections | Data structures |

## Privacy

- All audio and transcripts are processed and stored locally on device
- Network access is only required for initial model downloads
- No cloud transcription or analytics services are used

## License

Apache License 2.0. See `LICENSE`.

Model weights are downloaded at runtime and have their own licenses — see `NOTICE`.

## Creator

Created by **Akinori Nakajima** ([atyenoria](https://github.com/atyenoria)).
