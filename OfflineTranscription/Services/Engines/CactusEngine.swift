import Foundation
import CactusKit

@MainActor
final class CactusEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var loadingStatusMessage: String = ""

    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private let recorder = AudioRecorder()
    private let artifactDownloader = ArtifactDownloader()
    private var runtime: CactusRuntime?
    private var segmentIdCounter: Int = 0
    private var currentRequest: ArtifactDownloadRequest?

    func setupModel(_ model: ModelInfo) async throws {
        guard BackendCapabilities.isBackendSupported(.cactus) else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "CactusEngine",
                code: -10,
                userInfo: [NSLocalizedDescriptionKey: "Cactus backend is not supported on this platform"]
            ))
        }

        guard let request = try await makeDownloadRequest(for: model) else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "CactusEngine",
                code: -11,
                userInfo: [NSLocalizedDescriptionKey: "Missing Cactus runtime metadata for model \(model.id)"]
            ))
        }

        currentRequest = request
        modelState = .downloading
        loadingStatusMessage = "Downloading Cactus artifacts..."
        artifactDownloader.onProgress = { [weak self] value in
            self?.downloadProgress = value
        }

        let modelDirectory = try await artifactDownloader.downloadArtifacts(for: request)

        modelState = .loading
        loadingStatusMessage = "Loading Cactus runtime..."
        let runtime = CactusRuntime(config: CactusRuntimeConfig(modelDirectory: modelDirectory))
        do {
            try runtime.warmup()
            self.runtime = runtime
            self.modelState = .loaded
            self.downloadProgress = 1
            self.loadingStatusMessage = ""
        } catch {
            self.modelState = .error
            self.loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func loadModel(_ model: ModelInfo) async throws {
        guard let request = try await makeDownloadRequest(for: model) else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "CactusEngine",
                code: -12,
                userInfo: [NSLocalizedDescriptionKey: "Missing Cactus runtime metadata for model \(model.id)"]
            ))
        }

        guard artifactDownloader.areArtifactsPresent(for: request) else {
            modelState = .unloaded
            return
        }

        currentRequest = request
        modelState = .loading
        loadingStatusMessage = "Loading Cactus runtime..."

        let modelDirectory = artifactDownloader.localDirectory(for: request)
        let runtime = CactusRuntime(config: CactusRuntimeConfig(modelDirectory: modelDirectory))
        do {
            try runtime.warmup()
            self.runtime = runtime
            self.modelState = .loaded
            self.downloadProgress = 1
            self.loadingStatusMessage = ""
        } catch {
            self.modelState = .error
            self.loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        guard let request = currentRequestFor(model) else { return false }
        return artifactDownloader.areArtifactsPresent(for: request)
    }

    func unloadModel() async {
        stopRecording()
        runtime = nil
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
        guard let runtime else {
            throw AppError.modelNotReady
        }

        let text: String
        do {
            text = try runtime.transcribe(samples: audioArray, language: options.language)
        } catch {
            throw AppError.transcriptionFailed(underlying: error)
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ASRResult(text: "", segments: [], language: options.language)
        }

        let duration = Float(audioArray.count) / 16000
        let segment = ASRSegment(
            id: segmentIdCounter,
            text: " " + trimmed,
            start: 0,
            end: duration
        )
        segmentIdCounter += 1

        return ASRResult(text: trimmed, segments: [segment], language: options.language)
    }

    private func makeDownloadRequest(for model: ModelInfo) async throws -> ArtifactDownloadRequest? {
        guard let cardId = model.cardId,
              let runtimeVariantId = model.runtimeVariantId else {
            return nil
        }

        let catalog = await ModelCatalogService.shared.loadCatalog()
        guard let card = catalog.cards.first(where: { $0.id == cardId }),
              let variant = card.runtimeVariants.first(where: { $0.id == runtimeVariantId }) else {
            return nil
        }

        return ArtifactDownloadRequest(
            cardId: card.id,
            backend: variant.backend,
            version: variant.id,
            artifacts: variant.artifacts
        )
    }

    private func currentRequestFor(_ model: ModelInfo) -> ArtifactDownloadRequest? {
        if let currentRequest,
           currentRequest.cardId == model.cardId,
           currentRequest.backend == model.backend {
            return currentRequest
        }

        guard let cardId = model.cardId,
              let backend = model.backend,
              let runtimeVariantId = model.runtimeVariantId else {
            return nil
        }

        let catalog = ModelCatalogService.shared.loadLocalFallbackCatalog()
        guard let card = catalog.cards.first(where: { $0.id == cardId }),
              let variant = card.runtimeVariants.first(where: { $0.id == runtimeVariantId }),
              !variant.artifacts.isEmpty else {
            return nil
        }

        return ArtifactDownloadRequest(
            cardId: card.id,
            backend: backend,
            version: runtimeVariantId,
            artifacts: variant.artifacts
        )
    }
}
