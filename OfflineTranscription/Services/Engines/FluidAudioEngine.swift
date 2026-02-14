import Foundation
import FluidAudio

/// ASREngine implementation for FluidAudio (Parakeet-TDT, CoreML).
@MainActor
final class FluidAudioEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0.0
    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private var asrManager: AsrManager?
    private let recorder = AudioRecorder()
    private var segmentIdCounter: Int = 0

    private static let downloadedKey = "fluidAudio_downloaded_v3"

    /// Check if the current device supports FluidAudio's CoreML models.
    /// Requires A13 Bionic or newer (second-gen Neural Engine).
    /// A12X and older crash during CoreML inference with fatalError.
    nonisolated static var isDeviceSupported: Bool {
        #if os(macOS)
        // macOS: supported on Apple Silicon (arm64) only
        #if arch(arm64)
        return true
        #else
        return false
        #endif
        #else
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        let model = String(cString: machine)

        // iPad8,x = iPad Pro 3rd gen (A12X) — crashes
        // iPad11,x = iPad Air 3rd gen / iPad mini 5th gen (A12) — likely crashes
        // iPad7,x = iPad 6th gen (A10) — too old
        // All iPad models <= iPad8,x have A12X or older
        // iPad12,x (A13), iPad13,x (A14/M1), iPad14,x+ are supported
        if model.hasPrefix("iPad") {
            // Extract the major number: "iPad12,1" → 12
            let numPart = model.dropFirst(4) // drop "iPad"
            if let major = Int(numPart.prefix(while: { $0.isNumber })) {
                return major >= 12 // iPad12,x = iPad 9th gen (A13) and newer
            }
        }

        // iPhone12,x = iPhone 11 (A13) — supported
        // iPhone11,x = iPhone XS/XR (A12) — crashes
        if model.hasPrefix("iPhone") {
            let numPart = model.dropFirst(6) // drop "iPhone"
            if let major = Int(numPart.prefix(while: { $0.isNumber })) {
                return major >= 12 // iPhone12,x = iPhone 11 (A13) and newer
            }
        }

        // iPod — too old, not supported
        if model.hasPrefix("iPod") {
            return false
        }

        // Simulator or unknown — allow (will fail gracefully via FluidAudio's own checks)
        return true
        #endif
    }

    // MARK: - ASREngine

    func setupModel(_ model: ModelInfo) async throws {
        guard Self.isDeviceSupported else {
            modelState = .error
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "FluidAudioEngine", code: -10,
                userInfo: [NSLocalizedDescriptionKey: "Parakeet requires A13 chip or newer. This device is not supported."]
            ))
        }

        modelState = .downloading
        downloadProgress = 0.5 // FluidAudio manages its own download; show indeterminate progress

        do {
            let models = try await AsrModels.downloadAndLoad(version: .v3)
            modelState = .downloaded
            downloadProgress = 1.0
            UserDefaults.standard.set(true, forKey: Self.downloadedKey)

            modelState = .loading
            let manager = AsrManager(config: .default)
            try await manager.initialize(models: models)
            self.asrManager = manager
            modelState = .loaded
        } catch {
            modelState = .error
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func loadModel(_ model: ModelInfo) async throws {
        try await setupModel(model)
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        UserDefaults.standard.bool(forKey: Self.downloadedKey)
    }

    func unloadModel() async {
        recorder.stopRecording()
        asrManager = nil
        modelState = .unloaded
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        try await recorder.startRecording(captureMode: captureMode)
    }

    func stopRecording() {
        recorder.stopRecording()
    }

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        guard let asrManager else {
            throw AppError.modelNotReady
        }

        let result = try await asrManager.transcribe(audioArray, source: .system)
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else {
            return ASRResult(text: "", segments: [], language: nil)
        }

        let duration = Float(audioArray.count) / 16000.0
        let segId = segmentIdCounter
        segmentIdCounter += 1
        let segment = ASRSegment(
            id: segId,
            text: " " + text,
            start: 0,
            end: duration
        )

        return ASRResult(
            text: text,
            segments: [segment],
            language: options.language
        )
    }
}
