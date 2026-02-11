import AVFoundation
import Foundation
import WhisperKit

/// ASREngine implementation backed by WhisperKit (CoreML + Neural Engine).
/// Handles Whisper model download/load and recording via WhisperKit's AudioProcessor.
@MainActor
final class WhisperKitEngine: ASREngine {

    // MARK: - ASREngine conformance

    let isStreaming = false

    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0.0
    private(set) var loadingStatusMessage: String = ""

    private var whisperKit: WhisperKit?
    private var currentModelVariant: String?
    private var lastEnergyUpdateTime: CFAbsoluteTime = 0
    private var sessionStartSampleIndex: Int = 0
    private var sessionStartEnergyIndex: Int = 0

    // Cached values updated from the recording callback
    private(set) var audioSamples: [Float] = []
    private(set) var relativeEnergy: [Float] = []

    private func refreshSessionOffsets(using kit: WhisperKit) {
        sessionStartSampleIndex = kit.audioProcessor.audioSamples.count
        sessionStartEnergyIndex = kit.audioProcessor.relativeEnergy.count
    }

    private var preferredComputeOptions: ModelComputeOptions {
        #if targetEnvironment(simulator)
        // Simulator has no Neural Engine; GPU-backed compute is more stable for long decode.
        return ModelComputeOptions(
            audioEncoderCompute: .cpuAndGPU,
            textDecoderCompute: .cpuAndGPU
        )
        #else
        return ModelComputeOptions(
            audioEncoderCompute: .cpuAndNeuralEngine,
            textDecoderCompute: .cpuAndNeuralEngine
        )
        #endif
    }

    // MARK: - Model Management

    private func modelFolderKey(for variant: String) -> String {
        "modelFolder_\(variant)"
    }

    func setupModel(_ model: ModelInfo) async throws {
        let logger = InferenceLogger.shared
        guard let variant = model.variant else {
            throw AppError.noModelSelected
        }

        logger.log("[WhisperKitEngine] setupModel: variant=\(variant)")

        // Phase 1: Download
        modelState = .downloading
        downloadProgress = 0.0

        let modelFolderURL: URL
        do {
            modelFolderURL = try await WhisperKit.download(
                variant: variant,
                from: "argmaxinc/whisperkit-coreml",
                progressCallback: { [weak self] progress in
                    Task { @MainActor in
                        self?.downloadProgress = progress.fractionCompleted
                    }
                }
            )
            logger.log("[WhisperKitEngine] Download complete: \(modelFolderURL.path())")
        } catch {
            logger.log("[WhisperKitEngine] Download FAILED: \(error)")
            modelState = .unloaded
            downloadProgress = 0.0
            throw AppError.modelDownloadFailed(underlying: error)
        }

        modelState = .downloaded
        downloadProgress = 1.0

        // Phase 2: Load (CoreML compilation can take 30-60s on first use)
        modelState = .loading
        loadingStatusMessage = "Compiling CoreML model..."
        do {
            logger.log("[WhisperKitEngine] Loading model from: \(modelFolderURL.path())")
            let config = WhisperKitConfig(
                model: variant,
                modelFolder: modelFolderURL.path(),
                computeOptions: preferredComputeOptions,
                verbose: true,
                logLevel: .info,
                prewarm: true,
                load: true,
                download: false
            )

            whisperKit = try await WhisperKit(config)
            currentModelVariant = variant
            modelState = .loaded
            loadingStatusMessage = ""
            logger.log("[WhisperKitEngine] Model loaded successfully: \(variant)")

            UserDefaults.standard.set(
                modelFolderURL.path(),
                forKey: modelFolderKey(for: variant)
            )
        } catch {
            logger.log("[WhisperKitEngine] Load FAILED: \(error)")
            modelState = .unloaded
            downloadProgress = 0.0
            loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func loadModel(_ model: ModelInfo) async throws {
        guard let variant = model.variant else {
            throw AppError.noModelSelected
        }

        guard let savedFolder = UserDefaults.standard.string(
            forKey: modelFolderKey(for: variant)
        ), FileManager.default.fileExists(atPath: savedFolder) else {
            return
        }

        modelState = .loading
        loadingStatusMessage = "Compiling CoreML model..."

        do {
            let config = WhisperKitConfig(
                model: variant,
                modelFolder: savedFolder,
                computeOptions: preferredComputeOptions,
                verbose: true,
                logLevel: .info,
                prewarm: true,
                load: true,
                download: false
            )

            whisperKit = try await WhisperKit(config)
            currentModelVariant = variant
            modelState = .loaded
            loadingStatusMessage = ""
        } catch {
            currentModelVariant = nil
            modelState = .unloaded
            loadingStatusMessage = ""
        }
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        guard let variant = model.variant else { return false }
        guard let savedFolder = UserDefaults.standard.string(
            forKey: modelFolderKey(for: variant)
        ) else { return false }
        return FileManager.default.fileExists(atPath: savedFolder)
    }

    func unloadModel() async {
        stopRecording()
        await whisperKit?.unloadModels()
        whisperKit = nil
        currentModelVariant = nil
        modelState = .unloaded
        downloadProgress = 0.0
        audioSamples = []
        relativeEnergy = []
        sessionStartSampleIndex = 0
        sessionStartEnergyIndex = 0
    }

    // MARK: - Recording

    func startRecording(captureMode: AudioCaptureMode) async throws {
        guard let whisperKit else { throw AppError.modelNotReady }

        // Explicitly reset cached samples so a restarted session never reuses
        // stale buffers from a previous inference run.
        audioSamples = []
        relativeEnergy = []
        lastEnergyUpdateTime = 0
        refreshSessionOffsets(using: whisperKit)

        let granted = await AudioProcessor.requestRecordPermission()
        guard granted else { throw AppError.microphonePermissionDenied }

        try whisperKit.audioProcessor.startRecordingLive(inputDeviceID: nil) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                let now = CFAbsoluteTimeGetCurrent()
                guard now - self.lastEnergyUpdateTime > 0.1 else { return }
                self.lastEnergyUpdateTime = now
                if let kit = self.whisperKit {
                    let rawEnergy = kit.audioProcessor.relativeEnergy
                    let rawSamples = Array(kit.audioProcessor.audioSamples)
                    let energyStart = min(self.sessionStartEnergyIndex, rawEnergy.count)
                    let sampleStart = min(self.sessionStartSampleIndex, rawSamples.count)
                    self.relativeEnergy = Array(rawEnergy.dropFirst(energyStart))
                    self.audioSamples = Array(rawSamples.dropFirst(sampleStart))
                }
            }
        }
    }

    func stopRecording() {
        whisperKit?.audioProcessor.stopRecording()
        if let kit = whisperKit {
            refreshSessionOffsets(using: kit)
        } else {
            sessionStartSampleIndex = 0
            sessionStartEnergyIndex = 0
        }
        audioSamples = []
        relativeEnergy = []
    }

    // MARK: - Transcription

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        guard let whisperKit else { throw AppError.modelNotReady }
        let logger = InferenceLogger.shared

        // Keep worker fan-out conservative on older iPads to avoid decode stalls.
        let workerCount = min(2, max(1, ProcessInfo.processInfo.activeProcessorCount))
        let decodingOptions = DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: options.language,
            temperature: options.temperature,
            temperatureFallbackCount: 3,
            usePrefillPrompt: true,
            usePrefillCache: true,
            skipSpecialTokens: true,
            withoutTimestamps: !options.withTimestamps,
            wordTimestamps: options.withTimestamps,
            compressionRatioThreshold: 2.2,
            logProbThreshold: -1.0,
            noSpeechThreshold: 0.6,
            concurrentWorkerCount: workerCount
        )

        let results = try await whisperKit.transcribe(
            audioArray: audioArray,
            decodeOptions: decodingOptions
        )

        guard let result = results.first else {
            return ASRResult(text: "", segments: [], language: nil)
        }

        let primary = mapResult(result, timeOffset: 0, startingId: 0)
        let isLongAudio = audioArray.count >= 16000 * 18
        if !isLongAudio {
            return primary
        }

        let primaryScore = transcriptionQualityScore(primary.text)
        let isWhisperBaseVariant = (currentModelVariant ?? "").contains("whisper-base")
        // Only pay the chunked cost when primary output is empty or suspicious.
        let shouldRunChunkedFallback = isWhisperBaseVariant || primary.text.isEmpty || primaryScore < 140
        if !shouldRunChunkedFallback {
            return primary
        }

        // For low-quality long-audio output, run chunked decode and keep the better result.
        let chunked = try await transcribeChunked(
            whisperKit: whisperKit,
            audioArray: audioArray,
            decodingOptions: decodingOptions,
            aggressiveMode: isWhisperBaseVariant
        )
        if primary.text.isEmpty {
            return chunked
        }
        if chunked.text.isEmpty {
            return primary
        }
        let chunkedScore = transcriptionQualityScore(chunked.text)
        logger.log("[WhisperKitEngine] long-audio quality primaryScore=\(primaryScore) chunkedScore=\(chunkedScore) primary=\"\(String(primary.text.prefix(120)))\" chunked=\"\(String(chunked.text.prefix(120)))\"")
        if isWhisperBaseVariant {
            return chunkedScore >= primaryScore ? chunked : primary
        }
        return chunkedScore > primaryScore ? chunked : primary
    }

    private func mapResult(
        _ result: TranscriptionResult,
        timeOffset: Float,
        startingId: Int
    ) -> ASRResult {
        let segments = result.segments.enumerated().map { idx, seg in
            ASRSegment(
                id: startingId + idx,
                text: seg.text,
                start: seg.start + timeOffset,
                end: seg.end + timeOffset
            )
        }

        let segmentText = result.segments.map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let text = segmentText.isEmpty
            ? result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            : segmentText

        return ASRResult(text: text, segments: segments, language: result.language)
    }

    private func transcribeChunked(
        whisperKit: WhisperKit,
        audioArray: [Float],
        decodingOptions: DecodingOptions,
        aggressiveMode: Bool = false
    ) async throws -> ASRResult {
        let chunkSize = aggressiveMode ? 16000 * 5 : 16000 * 12
        let overlap = aggressiveMode ? 16000 / 2 : 16000 * 2
        var offset = 0
        var nextId = 0
        var combinedText: [String] = []
        var combinedSegments: [ASRSegment] = []
        var detectedLanguage: String?

        while offset < audioArray.count {
            let end = min(offset + chunkSize, audioArray.count)
            let chunk = Array(audioArray[offset..<end])
            let chunkOffsetSeconds = Float(offset) / 16000.0

            let chunkResults = try await whisperKit.transcribe(
                audioArray: chunk,
                decodeOptions: decodingOptions
            )
            if let chunkResult = chunkResults.first {
                let mapped = mapResult(
                    chunkResult,
                    timeOffset: chunkOffsetSeconds,
                    startingId: nextId
                )
                if !mapped.text.isEmpty {
                    if let last = combinedText.last {
                        if mapped.text == last {
                            // duplicate overlap decode, ignore
                        } else if mapped.text.hasPrefix(last) {
                            combinedText[combinedText.count - 1] = mapped.text
                        } else if last.hasPrefix(mapped.text) {
                            // keep the longer prior decode
                        } else {
                            combinedText.append(mapped.text)
                        }
                    } else {
                        combinedText.append(mapped.text)
                    }
                    combinedSegments.append(contentsOf: mapped.segments)
                    nextId += mapped.segments.count
                    if detectedLanguage == nil {
                        detectedLanguage = mapped.language
                    }
                }
            }

            if end == audioArray.count { break }
            offset = max(end - overlap, offset + 1)
        }

        return ASRResult(
            text: combinedText.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            segments: combinedSegments,
            language: detectedLanguage
        )
    }

    private func transcriptionQualityScore(_ text: String) -> Int {
        let tokens = text
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
        guard !tokens.isEmpty else { return Int.min }

        let uniqueTokenCount = Set(tokens).count
        var repeatedRunPenalty = 0
        var runLength = 1
        for idx in 1..<tokens.count {
            if tokens[idx] == tokens[idx - 1] {
                runLength += 1
            } else {
                if runLength > 2 {
                    repeatedRunPenalty += runLength - 2
                }
                runLength = 1
            }
        }
        if runLength > 2 {
            repeatedRunPenalty += runLength - 2
        }

        let alphaNumericCount = text.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count
        var score = 0
        score += tokens.count * 2
        score += uniqueTokenCount * 3
        score -= max(tokens.count - uniqueTokenCount, 0) * 2
        score -= repeatedRunPenalty * 8
        score += min(alphaNumericCount / 6, 40)
        if uniqueTokenCount <= 4 && tokens.count >= 12 {
            score -= 120
        }
        return score
    }
}
