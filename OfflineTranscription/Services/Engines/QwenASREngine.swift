import Foundation
import QwenASRKit

@MainActor
final class QwenASREngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var loadingStatusMessage: String = ""

    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private let recorder = AudioRecorder()
    private let downloader = ModelDownloader()
    private var qwen: QwenASR?
    private var segmentIdCounter: Int = 0

    func setupModel(_ model: ModelInfo) async throws {
        guard model.qwenModelConfig != nil else {
            throw AppError.noModelSelected
        }

        modelState = .downloading
        loadingStatusMessage = "Downloading Qwen ASR artifacts..."
        downloader.onProgress = { [weak self] value in
            self?.downloadProgress = value
        }

        let modelDir: URL
        do {
            modelDir = try await downloader.downloadModel(model)
        } catch {
            modelState = .error
            throw AppError.modelDownloadFailed(underlying: error)
        }

        modelState = .loading
        loadingStatusMessage = "Loading Qwen ASR runtime..."

        guard let runtime = QwenASR(modelDir: modelDir.path) else {
            modelState = .error
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "QwenASREngine",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize Qwen ASR runtime"]
            ))
        }

        qwen = runtime
        modelState = .loaded
        downloadProgress = 1
        loadingStatusMessage = ""
    }

    func loadModel(_ model: ModelInfo) async throws {
        guard model.qwenModelConfig != nil else {
            throw AppError.noModelSelected
        }
        guard downloader.isModelDownloaded(model),
              let modelDir = downloader.modelDirectory(for: model) else {
            modelState = .unloaded
            return
        }

        modelState = .loading
        loadingStatusMessage = "Loading Qwen ASR runtime..."

        guard let runtime = QwenASR(modelDir: modelDir.path) else {
            modelState = .error
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "QwenASREngine",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize Qwen ASR runtime"]
            ))
        }

        qwen = runtime
        modelState = .loaded
        loadingStatusMessage = ""
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        downloader.isModelDownloaded(model)
    }

    func unloadModel() async {
        stopRecording()
        qwen?.release()
        qwen = nil
        modelState = .unloaded
        downloadProgress = 0
        loadingStatusMessage = ""
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        try await recorder.startRecording(captureMode: captureMode)
    }

    func stopRecording() {
        recorder.stopRecording()
    }

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        guard let qwen else { throw AppError.modelNotReady }

        qwen.setLanguage(options.language)
        guard let text = qwen.transcribe(samples: audioArray)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            return ASRResult(text: "", segments: [], language: options.language)
        }

        let duration = Float(audioArray.count) / 16000
        let segment = ASRSegment(
            id: segmentIdCounter,
            text: " " + text,
            start: 0,
            end: duration
        )
        segmentIdCounter += 1

        return ASRResult(text: text, segments: [segment], language: options.language)
    }
}
