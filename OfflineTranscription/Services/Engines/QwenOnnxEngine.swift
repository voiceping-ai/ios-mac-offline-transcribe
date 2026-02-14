import Foundation

/// Compatibility shim for legacy `.qwenOnnx` engine selections.
///
/// The dedicated ONNX wrapper is currently disabled in `QwenASRKit`,
/// so this shim delegates to the maintained `QwenASREngine` backend.
@MainActor
final class QwenOnnxEngine: ASREngine {
    private let fallback = QwenASREngine()

    var isStreaming: Bool { fallback.isStreaming }
    var modelState: ASRModelState { fallback.modelState }
    var downloadProgress: Double { fallback.downloadProgress }
    var loadingStatusMessage: String { fallback.loadingStatusMessage }
    var audioSamples: [Float] { fallback.audioSamples }
    var relativeEnergy: [Float] { fallback.relativeEnergy }

    func setupModel(_ model: ModelInfo) async throws {
        try await fallback.setupModel(model)
    }

    func loadModel(_ model: ModelInfo) async throws {
        try await fallback.loadModel(model)
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        fallback.isModelDownloaded(model)
    }

    func unloadModel() async {
        await fallback.unloadModel()
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        try await fallback.startRecording(captureMode: captureMode)
    }

    func stopRecording() {
        fallback.stopRecording()
    }

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        try await fallback.transcribe(audioArray: audioArray, options: options)
    }
}
