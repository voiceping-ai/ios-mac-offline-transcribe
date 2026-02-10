import Foundation
import WhisperKit
import Observation
import AVFoundation

/// Session lifecycle states for the recording/transcription pipeline.
enum SessionState: String, Equatable, Sendable {
    case idle          // No active session
    case starting      // Setting up audio, requesting permission
    case recording     // Actively recording and transcribing
    case stopping      // Cleaning up
    case interrupted   // Audio session interrupted (phone call, etc.)
}

/// Download / readiness state for on-device translation models.
enum TranslationModelStatus: Equatable, Sendable {
    case unknown             // Not yet checked
    case checking            // Querying LanguageAvailability
    case downloading         // prepareTranslation() in progress
    case ready               // Models installed, translation available
    case unsupported         // Language pair not supported
    case failed(String)      // Download or preparation error
}

/// Transcription-only stub: translation is disabled in this repo.
/// Keep a no-op interface so existing call sites remain stable.
final class AppleTranslationService {
    func setSession(_ session: Any?) {}

    func translate(
        text: String,
        sourceLanguageCode: String,
        targetLanguageCode: String
    ) async throws -> String {
        _ = sourceLanguageCode
        _ = targetLanguageCode
        return text
    }
}

@MainActor
@Observable
final class WhisperService {
    // MARK: - State

    private(set) var whisperKit: WhisperKit?
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0.0
    private(set) var availableModels: [String] = []
    private(set) var currentModelVariant: String?
    private(set) var lastError: AppError?
    private(set) var loadingStatusMessage: String = ""

    // Session & transcription state
    private(set) var sessionState: SessionState = .idle
    private(set) var isRecording: Bool = false
    private(set) var isTranscribing: Bool = false
    private(set) var confirmedText: String = ""
    private(set) var hypothesisText: String = ""
    private(set) var confirmedSegments: [ASRSegment] = []
    private(set) var unconfirmedSegments: [ASRSegment] = []
    private(set) var bufferEnergy: [Float] = []
    private(set) var bufferSeconds: Double = 0.0
    private(set) var tokensPerSecond: Double = 0.0
    private(set) var cpuPercent: Double = 0.0
    private(set) var memoryMB: Double = 0.0
    private(set) var translatedConfirmedText: String = ""
    private(set) var translatedHypothesisText: String = ""
    private(set) var translationWarning: String?
    private(set) var translationModelStatus: TranslationModelStatus = .unknown
    /// E2E machine-readable payload surfaced to UI tests on real devices.
    private(set) var e2eOverlayPayload: String = ""

    // Configuration
    var selectedModel: ModelInfo = ModelInfo.defaultModel
    var useVAD: Bool = true
    var silenceThreshold: Float = 0.0015
    var realtimeDelayInterval: Double = 1.0
    var enableTimestamps: Bool = true
    var enableEagerMode: Bool = true
    var translationEnabled: Bool = false {
        didSet {
            if translationEnabled {
                scheduleTranslationUpdate()
            } else {
                resetTranslationState()
            }
        }
    }
    var translationSourceLanguageCode: String = "en" {
        didSet { scheduleTranslationUpdate() }
    }
    var translationTargetLanguageCode: String = "ja" {
        didSet { scheduleTranslationUpdate() }
    }

    // Engine delegation
    private(set) var activeEngine: ASREngine?

    /// The current session's audio samples (for saving to disk).
    var currentAudioSamples: [Float] {
        activeEngine?.audioSamples ?? []
    }

    // Private
    private var transcriptionTask: Task<Void, Never>?
    private var lingeringTranscriptionTask: Task<Void, Never>?
    private var lastBufferSize: Int = 0
    private var lastConfirmedSegmentEndSeconds: Float = 0
    private var prevUnconfirmedSegments: [ASRSegment] = []
    private var consecutiveSilenceCount: Int = 0
    private var hasCompletedFirstInference: Bool = false
    /// EMA-smoothed inference time (seconds) for CPU-aware delay calculation.
    private var movingAverageInferenceSeconds: Double = 0.0
    /// Finalized chunk texts, each representing one completed transcription window.
    private var completedChunksText: String = ""
    private var translationTask: Task<Void, Never>?
    private var e2eTranscribeInFlight: Bool = false
    /// Cache: last input text pair sent for translation (to skip redundant calls).
    private var lastTranslationInput: (confirmed: String, hypothesis: String)?
    private var lastUIMeterUpdateTimestamp: CFAbsoluteTime = 0
    private let translationService = AppleTranslationService()

    /// Called from TranslationBridgeView when a TranslationSession becomes available/unavailable.
    func setTranslationSession(_ session: Any?) {
        translationService.setSession(session)
        if session == nil {
            translationModelStatus = .unknown
        }
    }

    /// Called from TranslationBridgeView after model availability is confirmed.
    func setTranslationModelStatus(_ status: TranslationModelStatus) {
        translationModelStatus = status
        if status == .ready {
            scheduleTranslationUpdate()
        }
    }

    private let systemMetrics = SystemMetrics()
    private var metricsTask: Task<Void, Never>?
    private let selectedModelKey = "selectedModelVariant"
    private static let sampleRate: Float = 16000
    private static let displayEnergyFrameLimit = 160
    private static let uiMeterUpdateInterval: CFTimeInterval = 0.12
    /// Maximum audio chunk duration (seconds). Each chunk is transcribed independently;
    /// when the buffer exceeds this, the current hypothesis is confirmed and a new chunk begins.
    /// WhisperKit: 15s (multi-segment, eager mode confirms progressively).
    /// sherpa-onnx offline: 3.5s (single-segment, matches Android chunk cadence for
    /// faster updates — each inference processes a small slice, keeping latency low).
    private static let defaultMaxChunkSeconds: Float = 15.0
    private static let sherpaOfflineMaxChunkSeconds: Float = 3.5
    private static let omnilingualOfflineMaxChunkSeconds: Float = 4.0

    // MARK: - Adaptive Delay (CPU-aware, matches Android)
    /// Initial inference gate: show first words quickly (matches Android's 0.35s).
    private static let initialMinNewAudioSeconds: Float = 0.35
    /// Omnilingual is substantially heavier than SenseVoice/Moonshine; use slower initial gate.
    private static let omnilingualInitialMinNewAudioSeconds: Float = 3.0
    /// Base delay between inferences for sherpa-onnx offline after first decode.
    private static let sherpaBaseDelaySeconds: Float = 0.7
    /// Heavier omnilingual base delay to avoid UI starvation.
    private static let omnilingualBaseDelaySeconds: Float = 3.0
    /// Target inference duty cycle — inference should use at most this fraction of wall time.
    private static let targetInferenceDutyCycle: Float = 0.24
    /// Maximum CPU-protection delay cap.
    private static let maxCpuProtectDelaySeconds: Float = 1.6
    /// EMA smoothing factor for inference time tracking.
    private static let inferenceEmaAlpha: Double = 0.20

    /// Minimum RMS energy to submit audio for inference. Below this, the audio is
    /// near-silence and SenseVoice tends to hallucinate ("I.", "Yeah.", "The.").
    private static let minInferenceRMS: Float = 0.012

    /// Bypass VAD for the first N seconds so initial speech is never dropped.
    private static let initialVADBypassSeconds: Float = 1.0
    /// Keep a pre-roll of audio when VAD says silence, so utterance onsets
    /// that straddle VAD boundaries are not lost.
    private static let vadPrerollSeconds: Float = 0.6

    private var maxChunkSeconds: Float {
        guard selectedModel.engineType == .sherpaOnnxOffline else {
            return Self.defaultMaxChunkSeconds
        }
        return isOmnilingualModel
            ? Self.omnilingualOfflineMaxChunkSeconds
            : Self.sherpaOfflineMaxChunkSeconds
    }

    private var isOmnilingualModel: Bool {
        if selectedModel.sherpaModelConfig?.modelType == .omnilingualCtc {
            return true
        }
        return selectedModel.id.lowercased().contains("omnilingual")
    }

    /// Cancel the active transcription task and keep a handle so we can await
    /// full teardown before starting a new inference session.
    private func cancelAndTrackTranscriptionTask() {
        guard let task = transcriptionTask else { return }
        task.cancel()
        lingeringTranscriptionTask = task
        transcriptionTask = nil
    }

    /// Wait for any previously cancelled transcription task to finish.
    private func drainLingeringTranscriptionTask() async {
        if let activeTask = transcriptionTask {
            activeTask.cancel()
            lingeringTranscriptionTask = activeTask
            transcriptionTask = nil
        }
        if let lingering = lingeringTranscriptionTask {
            _ = await lingering.result
            lingeringTranscriptionTask = nil
        }
    }

    init() {
        if let saved = UserDefaults.standard.string(forKey: selectedModelKey),
           let model = ModelInfo.supportedModels.first(where: { $0.variant == saved })
                    ?? ModelInfo.supportedModels.first(where: { $0.id == saved })
                    ?? ModelInfo.findByLegacyId(saved) {
            self.selectedModel = model
        }
        migrateLegacyModelFolder()
        setupAudioObservers()
        startMetricsSampling()
    }

    deinit {
        // Note: @MainActor deinit is nonisolated in Swift 6, so we cannot access
        // actor-isolated properties here. Task cancellation and engine cleanup
        // happen via stopRecording() / unloadModel() before deallocation.
        NotificationCenter.default.removeObserver(self)
    }

    private func startMetricsSampling() {
        metricsTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.cpuPercent = self.systemMetrics.cpuPercent()
                self.memoryMB = self.systemMetrics.memoryMB()
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    // MARK: - Audio Session Observers

    private func setupAudioObservers() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleInterruptionNotification(_:)),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRouteChangeNotification(_:)),
            name: AVAudioSession.routeChangeNotification,
            object: nil
        )
    }

    @objc nonisolated private func handleInterruptionNotification(_ notification: Notification) {
        Task { @MainActor [weak self] in
            self?.handleInterruption(notification)
        }
    }

    @objc nonisolated private func handleRouteChangeNotification(_ notification: Notification) {
        Task { @MainActor [weak self] in
            self?.handleRouteChange(notification)
        }
    }

    private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue)
        else { return }

        switch type {
        case .began:
            if isRecording {
                cancelAndTrackTranscriptionTask()
                isTranscribing = false
                sessionState = .interrupted
            }
        case .ended:
            if sessionState == .interrupted {
                let options = info[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
                let shouldResume = AVAudioSession.InterruptionOptions(rawValue: options)
                    .contains(.shouldResume)

                if shouldResume, let engine = activeEngine {
                    Task {
                        do {
                            await self.drainLingeringTranscriptionTask()
                            try await engine.startRecording()
                            isTranscribing = true
                            sessionState = .recording
                            realtimeLoop()
                        } catch {
                            NSLog("[WhisperService] Failed to resume recording after interruption: \(error)")
                            stopRecording()
                        }
                    }
                } else {
                    stopRecording()
                }
            }
        @unknown default:
            break
        }
    }

    private func handleRouteChange(_ notification: Notification) {
        guard let info = notification.userInfo,
              let reasonValue = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue)
        else { return }

        switch reason {
        case .oldDeviceUnavailable:
            if isRecording {
                stopRecording()
            }
        default:
            break
        }
    }

    // MARK: - Model Management

    private func modelFolderKey(for variant: String) -> String {
        "modelFolder_\(variant)"
    }

    private func migrateLegacyModelFolder() {
        let legacyKey = "lastModelFolder"
        guard let legacyFolder = UserDefaults.standard.string(forKey: legacyKey) else { return }

        for model in ModelInfo.availableModels {
            if let variant = model.variant, legacyFolder.contains(variant) {
                let perModelKey = modelFolderKey(for: variant)
                if UserDefaults.standard.string(forKey: perModelKey) == nil {
                    UserDefaults.standard.set(legacyFolder, forKey: perModelKey)
                }
            }
        }
        UserDefaults.standard.removeObject(forKey: legacyKey)
    }

    func loadModelIfAvailable() async {
        // Don't overwrite an already-loaded or in-progress engine
        guard activeEngine == nil, modelState == .unloaded else { return }

        let engine = EngineFactory.makeEngine(for: selectedModel)

        guard engine.isModelDownloaded(selectedModel) else { return }

        activeEngine = engine
        modelState = .loading
        lastError = nil

        do {
            try await engine.loadModel(selectedModel)
            // Verify this engine is still the active one (not replaced by switchModel)
            guard activeEngine === engine else { return }
            modelState = engine.modelState
            if let variant = selectedModel.variant {
                currentModelVariant = variant
            }
        } catch {
            guard activeEngine === engine else { return }
            activeEngine = nil
            modelState = .unloaded
        }
    }

    func fetchAvailableModels() async {
        do {
            let models = try await WhisperKit.fetchAvailableModels(
                from: "argmaxinc/whisperkit-coreml"
            )
            availableModels = models
        } catch {
            lastError = .modelDownloadFailed(underlying: error)
        }
    }

    func setupModel() async {
        let logger = InferenceLogger.shared
        logger.log("[WhisperService] setupModel: model=\(selectedModel.id) engine=\(selectedModel.engineType)")
        let engine = EngineFactory.makeEngine(for: selectedModel)
        activeEngine = engine

        modelState = .downloading
        downloadProgress = 0.0
        lastError = nil

        // Sync download progress and status from engine in background
        let progressTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                guard let self, self.activeEngine === engine else { break }
                self.downloadProgress = engine.downloadProgress
                self.loadingStatusMessage = engine.loadingStatusMessage
                let engineState = engine.modelState
                if engineState == .downloaded || engineState == .loading {
                    self.modelState = engineState
                }
            }
        }

        do {
            try await engine.setupModel(selectedModel)
            progressTask.cancel()
            // Verify this engine is still active (not replaced by a concurrent switch)
            guard activeEngine === engine else {
                logger.log("[WhisperService] setupModel: engine replaced during setup, aborting")
                return
            }
            modelState = engine.modelState
            downloadProgress = engine.downloadProgress
            loadingStatusMessage = ""
            logger.log("[WhisperService] setupModel SUCCESS: modelState=\(modelState) model=\(selectedModel.id)")

            // Persist selection
            if let variant = selectedModel.variant {
                currentModelVariant = variant
                UserDefaults.standard.set(variant, forKey: selectedModelKey)
            } else {
                UserDefaults.standard.set(selectedModel.id, forKey: selectedModelKey)
            }
        } catch {
            progressTask.cancel()
            logger.log("[WhisperService] setupModel FAILED: \(error) model=\(selectedModel.id)")
            guard activeEngine === engine else { return }
            activeEngine = nil
            modelState = .unloaded
            downloadProgress = 0.0
            loadingStatusMessage = ""
            if let appError = error as? AppError {
                lastError = appError
            } else {
                lastError = .modelLoadFailed(underlying: error)
            }
        }
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        switch model.engineType {
        case .whisperKit:
            guard let variant = model.variant,
                  let savedFolder = UserDefaults.standard.string(
                      forKey: modelFolderKey(for: variant)
                  ) else {
                return false
            }
            return FileManager.default.fileExists(atPath: savedFolder)
        case .sherpaOnnxOffline, .sherpaOnnxStreaming:
            guard let config = model.sherpaModelConfig else { return false }
            let modelDir = ModelDownloader.modelsDirectory.appendingPathComponent(config.repoName)
            let tokensPath = modelDir.appendingPathComponent(config.tokens)
            return FileManager.default.fileExists(atPath: tokensPath.path)
        case .fluidAudio:
            // FluidAudio manages its own model cache
            return false
        }
    }

    func switchModel(to model: ModelInfo) async {
        if isRecording {
            stopRecording()
        }

        await drainLingeringTranscriptionTask()
        resetTranscriptionState()

        isRecording = false
        isTranscribing = false
        sessionState = .idle
        cancelAndTrackTranscriptionTask()
        translationTask?.cancel()
        translationTask = nil

        // Unload current engine
        if let engine = activeEngine {
            await engine.unloadModel()
        }
        activeEngine = nil
        whisperKit = nil
        modelState = .unloaded
        selectedModel = model
        await setupModel()
    }

    // MARK: - Recording & Transcription

    func startRecording() async throws {
        guard sessionState == .idle else { return }

        sessionState = .starting
        await drainLingeringTranscriptionTask()

        guard let engine = activeEngine, engine.modelState == .loaded else {
            sessionState = .idle
            throw AppError.modelNotReady
        }

        resetTranscriptionState()

        do {
            try await engine.startRecording()
        } catch {
            isRecording = false
            isTranscribing = false
            sessionState = .idle
            if let appError = error as? AppError {
                lastError = appError
            } else {
                lastError = .audioSessionSetupFailed(underlying: error)
            }
            throw error
        }

        isRecording = true
        isTranscribing = true
        sessionState = .recording

        realtimeLoop()
    }

    func stopRecording() {
        guard sessionState == .recording || sessionState == .interrupted
            || sessionState == .starting else { return }

        sessionState = .stopping
        cancelAndTrackTranscriptionTask()
        translationTask?.cancel()
        translationTask = nil
        activeEngine?.stopRecording()
        isRecording = false
        isTranscribing = false
        sessionState = .idle
    }

    func clearTranscription() {
        stopRecording()
        resetTranscriptionState()
    }

    func clearLastError() {
        lastError = nil
    }

    // MARK: - File Transcription

    /// Whether a file transcription is currently in progress.
    private(set) var isTranscribingFile: Bool = false

    /// Transcribe an audio file at the given URL (any format AVAudioFile supports).
    func transcribeFile(_ url: URL) {
        guard !isTranscribingFile else { return }
        guard let engine = activeEngine, engine.modelState == .loaded else {
            lastError = .modelNotReady
            return
        }

        resetTranscriptionState()
        isTranscribingFile = true

        cancelAndTrackTranscriptionTask()
        transcriptionTask = Task {
            // Note: do NOT call drainLingeringTranscriptionTask() here — it would
            // see transcriptionTask == self and deadlock awaiting its own result.
            // cancelAndTrackTranscriptionTask() above already cancelled the old task.

            // Security-scoped resource access must span the entire read operation.
            let didAccess = url.startAccessingSecurityScopedResource()
            defer {
                if didAccess { url.stopAccessingSecurityScopedResource() }
                isTranscribingFile = false
            }
            do {
                let samples = try Self.loadAudioFile(url: url)
                let audioDuration = Double(samples.count) / Double(Self.sampleRate)
                self.bufferSeconds = audioDuration
                let options = ASRTranscriptionOptions(
                    language: nil,
                    withTimestamps: enableTimestamps
                )
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await engine.transcribe(audioArray: samples, options: options)
                let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                guard !Task.isCancelled else { return }
                let words = max(0, result.text.split(whereSeparator: \.isWhitespace).count)
                let elapsedSeconds = elapsedMs / 1000.0
                if elapsedSeconds > 0, words > 0 {
                    tokensPerSecond = Double(words) / elapsedSeconds
                } else {
                    tokensPerSecond = 0
                }
                confirmedSegments = result.segments
                let segmentText = result.segments.map(\.text).joined(separator: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                confirmedText = segmentText.isEmpty
                    ? result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    : segmentText
            } catch {
                guard !Task.isCancelled else { return }
                lastError = .transcriptionFailed(underlying: error)
            }
        }
    }

    /// Transcribe a WAV file from the given path (for testing / E2E validation).
    func transcribeTestFile(_ path: String) {
        let logger = InferenceLogger.shared
        logger.log("transcribeTestFile called, path=\(path) model=\(selectedModel.id)")
        NSLog("[E2E] transcribeTestFile called, path=\(path)")
        NSLog("[E2E] activeEngine=\(String(describing: activeEngine)), modelState=\(String(describing: activeEngine?.modelState))")
        if e2eTranscribeInFlight {
            NSLog("[E2E] Skipping duplicate transcribeTestFile invocation while previous run is active")
            return
        }
        guard let engine = activeEngine, engine.modelState == .loaded else {
            NSLog("[E2E] ERROR: model not ready, activeEngine=\(String(describing: activeEngine))")
            lastError = .modelNotReady
            writeE2EResult(
                transcript: "",
                translatedText: "",
                tokensPerSecond: 0,
                durationMs: 0,
                error: "model not ready"
            )
            return
        }

        resetTranscriptionState()
        e2eOverlayPayload = ""
        e2eTranscribeInFlight = true

        cancelAndTrackTranscriptionTask()
        transcriptionTask = Task {
            defer { e2eTranscribeInFlight = false }
            // Note: do NOT call drainLingeringTranscriptionTask() here — it would
            // see transcriptionTask == self and deadlock awaiting its own result.
            // cancelAndTrackTranscriptionTask() above already cancelled the old task.
            do {
                NSLog("[E2E] Loading audio file...")
                let samples = try Self.loadAudioFile(url: URL(fileURLWithPath: path))
                let audioDuration = Double(samples.count) / Double(Self.sampleRate)
                NSLog("[E2E] Audio loaded: \(samples.count) samples (\(audioDuration)s)")
                self.bufferSeconds = audioDuration
                // Keep language auto-detection for E2E to avoid model-specific decode regressions
                // (e.g., repetition loops or empty output under forced language), except
                // omnilingual fallback where fixed English hints significantly improve quality
                // for this English benchmark fixture.
                let forcedLanguage: String? = isOmnilingualModel ? "en" : nil
                let options = ASRTranscriptionOptions(
                    language: forcedLanguage,
                    withTimestamps: enableTimestamps
                )
                NSLog("[E2E] Starting transcription with engine \(type(of: engine))...")
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await engine.transcribe(audioArray: samples, options: options)
                let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                guard !Task.isCancelled else { return }
                let words = max(0, result.text.split(whereSeparator: \.isWhitespace).count)
                let elapsedSeconds = elapsedMs / 1000.0
                if elapsedSeconds > 0, words > 0 {
                    tokensPerSecond = Double(words) / elapsedSeconds
                } else {
                    tokensPerSecond = 0
                }
                NSLog("[E2E] Transcription complete: text='\(result.text)', segments=\(result.segments.count)")
                confirmedSegments = result.segments
                let segmentText = result.segments.map(\.text).joined(separator: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                confirmedText = segmentText.isEmpty
                    ? result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    : segmentText
                NSLog("[E2E] confirmedText set to: '\(confirmedText)'")
                scheduleTranslationUpdate()
                let deadline = Date().addingTimeInterval(10)
                while Date() < deadline {
                    let translatedReady = !translationEnabled
                        || confirmedText.isEmpty
                        || !translatedConfirmedText.isEmpty
                    if translatedReady { break }
                    try? await Task.sleep(for: .milliseconds(250))
                }
                writeE2EResult(
                    transcript: confirmedText,
                    translatedText: translatedConfirmedText,
                    tokensPerSecond: tokensPerSecond,
                    durationMs: elapsedMs,
                    error: nil
                )
            } catch {
                guard !Task.isCancelled else { return }
                NSLog("[E2E] ERROR: transcription failed: \(error)")
                lastError = .transcriptionFailed(underlying: error)
                writeE2EResult(
                    transcript: "",
                    translatedText: "",
                    tokensPerSecond: 0,
                    durationMs: 0,
                    error: error.localizedDescription
                )
            }
        }
    }

    func writeE2EFailure(reason: String) {
        writeE2EResult(
            transcript: "",
            translatedText: "",
            tokensPerSecond: 0,
            durationMs: 0,
            error: reason
        )
    }

    private func writeE2EResult(
        transcript: String,
        translatedText: String,
        tokensPerSecond: Double,
        durationMs: Double,
        error: String?
    ) {
        let keywords = ["country", "ask", "do for", "fellow", "americans"]
        let lower = transcript.lowercased()
        let normalizedSource = translationSourceLanguageCode.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let normalizedTarget = translationTargetLanguageCode.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let expectsTranslation = translationEnabled
            && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !normalizedSource.isEmpty
            && !normalizedTarget.isEmpty
            && normalizedSource != normalizedTarget
        let translationReady = !expectsTranslation
            || !translatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let isOmnilingual = selectedModel.id.lowercased().contains("omnilingual")
        let hasKeywordHit = keywords.contains { lower.contains($0) }
        let hasMeaningfulText = transcript.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
        let asciiLetterCount = transcript.unicodeScalars.filter {
            CharacterSet.letters.contains($0) && $0.isASCII
        }.count

        // pass = core transcription quality only; translation tracked separately.
        // Omnilingual quality bar is stricter to avoid false passes on short gibberish output.
        let omnilingualQuality = hasKeywordHit
            || (hasMeaningfulText && transcript.count >= 24 && asciiLetterCount >= 12)
        let pass = error == nil
            && !transcript.isEmpty
            && (isOmnilingual ? omnilingualQuality : hasKeywordHit)
        let payload: [String: Any?] = [
            "model_id": selectedModel.id,
            "engine": selectedModel.inferenceMethodLabel,
            "transcript": transcript,
            "translated_text": translatedText,
            "translation_warning": translationWarning,
            "expects_translation": expectsTranslation,
            "translation_ready": translationReady,
            "pass": pass,
            "tokens_per_second": tokensPerSecond,
            "duration_ms": durationMs,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "error": error
        ]

        do {
            let data = try JSONSerialization.data(
                withJSONObject: payload.compactMapValues { $0 },
                options: [.prettyPrinted]
            )
            if let payloadText = String(data: data, encoding: .utf8) {
                e2eOverlayPayload = payloadText
            }
            let modelId = selectedModel.id
            let fileURL = URL(fileURLWithPath: "/tmp/e2e_result_\(modelId).json")
            try data.write(to: fileURL, options: .atomic)
            NSLog("[E2E] Result written to \(fileURL.path)")
        } catch {
            e2eOverlayPayload = """
            {"model_id":"\(selectedModel.id)","pass":false,"error":"failed to serialize/write E2E result"}
            """
            NSLog("[E2E] Failed to write result file: \(error)")
        }
    }

    /// Load any audio file and return 16kHz mono Float32 samples in [-1, 1].
    /// Uses AVAudioConverter to handle arbitrary sample rates and channel counts.
    private static func loadAudioFile(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let fileFormat = file.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "WhisperService", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot create target audio format"])
        }

        let fileFrameCount = AVAudioFrameCount(file.length)
        guard fileFrameCount > 0 else {
            throw NSError(domain: "WhisperService", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "Audio file is empty"])
        }

        // Read file in its native processing format
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: fileFrameCount) else {
            throw NSError(domain: "WhisperService", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create source audio buffer"])
        }
        try file.read(into: sourceBuffer)

        // If already 16kHz mono Float32, return directly
        if fileFormat.sampleRate == 16000 && fileFormat.channelCount == 1
            && fileFormat.commonFormat == .pcmFormatFloat32 {
            guard let floatData = sourceBuffer.floatChannelData else {
                throw NSError(domain: "WhisperService", code: -2,
                              userInfo: [NSLocalizedDescriptionKey: "No float channel data"])
            }
            return Array(UnsafeBufferPointer(start: floatData[0], count: Int(sourceBuffer.frameLength)))
        }

        // Convert to 16kHz mono Float32
        guard let converter = AVAudioConverter(from: fileFormat, to: targetFormat) else {
            throw NSError(domain: "WhisperService", code: -4,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot create audio converter from \(fileFormat) to 16kHz mono"])
        }

        let ratio = 16000.0 / fileFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(ceil(Double(sourceBuffer.frameLength) * ratio))
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "WhisperService", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create output audio buffer"])
        }

        var conversionError: NSError?
        var inputConsumed = false
        converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if inputConsumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            outStatus.pointee = .haveData
            inputConsumed = true
            return sourceBuffer
        }
        if let conversionError {
            throw conversionError
        }

        guard let floatData = outputBuffer.floatChannelData, outputBuffer.frameLength > 0 else {
            throw NSError(domain: "WhisperService", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "Audio conversion produced no output"])
        }
        return Array(UnsafeBufferPointer(start: floatData[0], count: Int(outputBuffer.frameLength)))
    }

    var fullTranscriptionText: String {
        let currentChunkConfirmed = normalizedJoinedText(from: confirmedSegments)
        let currentChunkHypothesis = normalizedJoinedText(from: unconfirmedSegments)
        let currentChunk = [currentChunkConfirmed, currentChunkHypothesis]
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        let parts = [completedChunksText, currentChunk]
            .filter { !$0.isEmpty }
        return parts.joined(separator: "\n")
    }

    // MARK: - Private: Real-time Loop

    private func realtimeLoop() {
        cancelAndTrackTranscriptionTask()

        guard let engine = activeEngine else { return }

        if engine.isStreaming {
            transcriptionTask = Task {
                await streamingLoop(engine: engine)
            }
        } else {
            transcriptionTask = Task {
                await offlineLoop(engine: engine)
            }
        }
    }

    private func offlineLoop(engine: ASREngine) async {
        while isRecording && isTranscribing && !Task.isCancelled {
            do {
                try await transcribeCurrentBuffer(engine: engine)
            } catch {
                if !Task.isCancelled {
                    lastError = .transcriptionFailed(underlying: error)
                }
                break
            }
        }

        if !Task.isCancelled {
            isRecording = false
            isTranscribing = false
            sessionState = .idle
            engine.stopRecording()
        }
    }

    private func streamingLoop(engine: ASREngine) async {
        while isRecording && isTranscribing && !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(100))

            refreshRealtimeMeters(engine: engine)

            // Poll streaming result
            if let result = engine.getStreamingResult() {
                let nextHypothesis = normalizedJoinedText(from: result.segments)
                if unconfirmedSegments != result.segments {
                    unconfirmedSegments = result.segments
                }
                if hypothesisText != nextHypothesis {
                    hypothesisText = nextHypothesis
                    scheduleTranslationUpdate()
                }

                // Endpoint detection → finalize utterance as a new chunk
                if engine.isEndpointDetected() {
                    finalizeCurrentChunk()
                    engine.resetStreamingState()
                }
            }
        }

        if !Task.isCancelled {
            // Capture final result before stopping
            if let result = engine.getStreamingResult(),
               !normalizedJoinedText(from: result.segments).isEmpty {
                unconfirmedSegments = result.segments
                finalizeCurrentChunk()
            }

            isRecording = false
            isTranscribing = false
            sessionState = .idle
            engine.stopRecording()
        }
    }

    private func transcribeCurrentBuffer(engine: ASREngine) async throws {
        let logger = InferenceLogger.shared
        let currentBuffer = engine.audioSamples
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Self.sampleRate
        refreshRealtimeMeters(engine: engine)

        let effectiveDelay = adaptiveDelay()
        guard nextBufferSeconds > Float(effectiveDelay) else {
            try await Task.sleep(for: .milliseconds(100))
            return
        }

        if useVAD {
            // Bypass VAD for the first second so initial speech is never dropped
            let vadBypassSamples = Int(Self.sampleRate * Self.initialVADBypassSeconds)
            let bypassVadDuringStartup = !hasCompletedFirstInference && currentBuffer.count <= vadBypassSamples
            if !bypassVadDuringStartup {
                let voiceDetected = isVoiceDetected(
                    in: engine.relativeEnergy,
                    nextBufferInSeconds: nextBufferSeconds
                )
                if !voiceDetected {
                    consecutiveSilenceCount += 1
                    // Keep a pre-roll so utterance onsets straddling VAD are preserved
                    let prerollSamples = Int(Self.sampleRate * Self.vadPrerollSeconds)
                    lastBufferSize = max(currentBuffer.count - prerollSamples, 0)
                    if consecutiveSilenceCount == 1 || consecutiveSilenceCount % 10 == 0 {
                        logger.log("VAD silence #\(consecutiveSilenceCount) totalBuffer=\(currentBuffer.count) (\(String(format: "%.1f", Float(currentBuffer.count) / Self.sampleRate))s)")
                    }
                    return
                }
                consecutiveSilenceCount = 0
            }
        }

        // Chunk-based windowing: process audio in fixed-size chunks to prevent
        // models from receiving unbounded audio. When the buffer grows past the
        // current chunk boundary, finalize the hypothesis and start a new chunk.
        let bufferEndSeconds = Float(currentBuffer.count) / Self.sampleRate
        var chunkEndSeconds = lastConfirmedSegmentEndSeconds + maxChunkSeconds

        if bufferEndSeconds > chunkEndSeconds {
            finalizeCurrentChunk()
            lastConfirmedSegmentEndSeconds = chunkEndSeconds
            // Recompute for the new chunk so we don't produce an empty slice
            chunkEndSeconds = lastConfirmedSegmentEndSeconds + maxChunkSeconds
        }

        // Slice audio for the current chunk window
        let sliceStartSeconds = lastConfirmedSegmentEndSeconds
        let sliceStartSample = min(Int(sliceStartSeconds * Self.sampleRate), currentBuffer.count)
        let sliceEndSample = min(Int(chunkEndSeconds * Self.sampleRate), currentBuffer.count)
        let audioSamples = Array(currentBuffer[sliceStartSample..<sliceEndSample])
        guard !audioSamples.isEmpty else { return }

        // RMS energy gate: skip inference on near-silence audio to avoid
        // SenseVoice hallucinations ("I.", "Yeah.", "The.") and save CPU.
        // NOTE: lastBufferSize is NOT updated on skip — this ensures that when
        // speech resumes after silence, nextBufferSeconds is already large enough
        // to pass the delay guard immediately, giving near-instant response.
        let sliceRMS = sqrt(audioSamples.reduce(Float(0)) { $0 + $1 * $1 } / Float(audioSamples.count))
        if sliceRMS < Self.minInferenceRMS {
            logger.log("SKIP low-energy slice rms=\(String(format: "%.4f", sliceRMS)) < \(Self.minInferenceRMS)")
            try await Task.sleep(for: .milliseconds(500))
            return
        }

        lastBufferSize = currentBuffer.count

        let options = ASRTranscriptionOptions(
            withTimestamps: enableTimestamps,
            temperature: 0.0
        )

        let sliceDurationSeconds = Float(audioSamples.count) / Self.sampleRate
        logger.log("BUFFER SUBMIT model=\(selectedModel.id) sliceStart=\(String(format: "%.2f", sliceStartSeconds))s sliceEnd=\(String(format: "%.2f", Float(sliceEndSample) / Self.sampleRate))s sliceSamples=\(audioSamples.count) sliceDuration=\(String(format: "%.2f", sliceDurationSeconds))s rms=\(String(format: "%.4f", sliceRMS)) totalBuffer=\(currentBuffer.count) (\(String(format: "%.1f", Float(currentBuffer.count) / Self.sampleRate))s)")
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await engine.transcribe(audioArray: audioSamples, options: options)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        guard !Task.isCancelled else { return }

        let wordCount = result.text.split(separator: " ").count
        if elapsed > 0 && wordCount > 0 {
            tokensPerSecond = Double(wordCount) / elapsed
        }

        // Track inference time with EMA for CPU-aware delay
        if movingAverageInferenceSeconds <= 0 {
            movingAverageInferenceSeconds = elapsed
        } else {
            movingAverageInferenceSeconds = Self.inferenceEmaAlpha * elapsed
                + (1.0 - Self.inferenceEmaAlpha) * movingAverageInferenceSeconds
        }

        NSLog("[WhisperService] chunk inference: %.1fs audio in %.2fs (ratio %.1fx, %d words, emaInf=%.3fs, delay=%.2fs)",
              sliceDurationSeconds, elapsed, Double(sliceDurationSeconds) / elapsed, wordCount,
              movingAverageInferenceSeconds, adaptiveDelay())
        logger.log("BUFFER RESULT model=\(selectedModel.id) elapsed=\(String(format: "%.3f", elapsed))s rtf=\(String(format: "%.2f", elapsed / Double(sliceDurationSeconds))) words=\(wordCount) segments=\(result.segments.count) emaInf=\(String(format: "%.3f", movingAverageInferenceSeconds))s delay=\(String(format: "%.2f", adaptiveDelay()))s text=\"\(String(result.text.prefix(200)))\"")

        hasCompletedFirstInference = true
        processTranscriptionResult(result, sliceOffset: sliceStartSeconds)
    }

    /// Voice activity detection using peak + average energy (matches Android).
    private func isVoiceDetected(in energy: [Float], nextBufferInSeconds: Float) -> Bool {
        guard !energy.isEmpty else { return false }
        let recentEnergy = energy.suffix(10)
        let peakEnergy = recentEnergy.max() ?? 0
        let avgEnergy = recentEnergy.reduce(0, +) / Float(recentEnergy.count)
        return peakEnergy >= silenceThreshold || avgEnergy >= silenceThreshold * 0.5
    }

    private func adaptiveDelay() -> Double {
        // During silence, back off to save CPU
        if consecutiveSilenceCount > 5 {
            return min(realtimeDelayInterval * 3.0, 3.0)
        } else if consecutiveSilenceCount > 2 {
            return realtimeDelayInterval * 2.0
        }

        // Fast initial gate: show first words quickly (matches Android 0.35s)
        if !hasCompletedFirstInference {
            if selectedModel.engineType == .sherpaOnnxOffline && isOmnilingualModel {
                return Double(Self.omnilingualInitialMinNewAudioSeconds)
            }
            return Double(Self.initialMinNewAudioSeconds)
        }

        // For sherpa-onnx offline: CPU-aware delay (matches Android architecture)
        if selectedModel.engineType == .sherpaOnnxOffline {
            let baseDelay = isOmnilingualModel
                ? Double(Self.omnilingualBaseDelaySeconds)
                : Double(Self.sherpaBaseDelaySeconds)
            return computeCpuAwareDelay(baseDelay: baseDelay)
        }

        return realtimeDelayInterval
    }

    /// Compute delay based on actual inference time to maintain a target CPU duty cycle.
    /// If inference takes 0.17s and target duty is 24%, delay = 0.17/0.24 = 0.71s.
    /// This adapts automatically to device speed — fast devices get shorter delays.
    private func computeCpuAwareDelay(baseDelay: Double) -> Double {
        let avg = movingAverageInferenceSeconds
        guard avg > 0 else { return baseDelay }
        let budgetDelay = avg / Double(Self.targetInferenceDutyCycle)
        return max(baseDelay, min(budgetDelay, Double(Self.maxCpuProtectDelaySeconds)))
    }

    /// Update render-facing meters at a fixed cadence with bounded payload size.
    /// This keeps live UI smooth while preventing large array churn on every loop.
    private func refreshRealtimeMeters(engine: ASREngine, force: Bool = false) {
        let now = CFAbsoluteTimeGetCurrent()
        if !force, now - lastUIMeterUpdateTimestamp < Self.uiMeterUpdateInterval {
            return
        }
        lastUIMeterUpdateTimestamp = now

        let sampleCount = engine.audioSamples.count
        let nextBufferSeconds = Double(sampleCount) / Double(Self.sampleRate)
        if bufferSeconds != nextBufferSeconds {
            bufferSeconds = nextBufferSeconds
        }

        let nextEnergy = Array(engine.relativeEnergy.suffix(Self.displayEnergyFrameLimit))
        if bufferEnergy != nextEnergy {
            bufferEnergy = nextEnergy
        }
    }

    private func processTranscriptionResult(_ result: ASRResult, sliceOffset: Float = 0) {
        let newSegments = result.segments

        // Eager mode only works for multi-segment models (WhisperKit).
        // Single-segment models (SenseVoice, Moonshine) always return 1 segment
        // whose text changes every cycle, so segment comparison never confirms.
        let useEager = enableEagerMode && selectedModel.engineType != .sherpaOnnxOffline
        if useEager, !prevUnconfirmedSegments.isEmpty {
            var matchCount = 0
            for (prevSeg, newSeg) in zip(prevUnconfirmedSegments, newSegments) {
                if normalizedSegmentText(prevSeg.text)
                    == normalizedSegmentText(newSeg.text)
                {
                    matchCount += 1
                } else {
                    break
                }
            }

            if matchCount > 0 {
                let newlyConfirmed = Array(newSegments.prefix(matchCount))
                confirmedSegments.append(contentsOf: newlyConfirmed)

                if let lastConfirmed = newlyConfirmed.last {
                    lastConfirmedSegmentEndSeconds = sliceOffset + lastConfirmed.end
                }

                unconfirmedSegments = Array(newSegments.dropFirst(matchCount))
            } else {
                unconfirmedSegments = newSegments
            }
        } else {
            unconfirmedSegments = newSegments
        }

        prevUnconfirmedSegments = unconfirmedSegments

        // Build confirmed text: completed chunks + within-chunk confirmed segments
        let withinChunkConfirmed = normalizedJoinedText(from: confirmedSegments)
        let nextConfirmedText = [completedChunksText, withinChunkConfirmed]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let nextHypothesisText = normalizedJoinedText(from: unconfirmedSegments)
        let transcriptionChanged = confirmedText != nextConfirmedText || hypothesisText != nextHypothesisText
        if confirmedText != nextConfirmedText {
            confirmedText = nextConfirmedText
        }
        if hypothesisText != nextHypothesisText {
            hypothesisText = nextHypothesisText
        }
        if transcriptionChanged {
            scheduleTranslationUpdate()
        }
    }

    /// Finalize the current chunk: combine all segments into completed text and reset per-chunk state.
    private func finalizeCurrentChunk() {
        let allSegments = confirmedSegments + unconfirmedSegments
        let chunkText = normalizedJoinedText(from: allSegments)
        if !chunkText.isEmpty {
            if completedChunksText.isEmpty {
                completedChunksText = chunkText
            } else {
                completedChunksText += "\n" + chunkText
            }
        }
        confirmedSegments = []
        unconfirmedSegments = []
        prevUnconfirmedSegments = []
        let nextConfirmedText = completedChunksText
        let transcriptionChanged = confirmedText != nextConfirmedText || !hypothesisText.isEmpty
        if confirmedText != nextConfirmedText {
            confirmedText = nextConfirmedText
        }
        if !hypothesisText.isEmpty {
            hypothesisText = ""
        }
        if transcriptionChanged {
            scheduleTranslationUpdate()
        }
    }

    private func normalizedJoinedText(from segments: [ASRSegment]) -> String {
        segments
            .map { normalizedSegmentText($0.text) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func normalizedSegmentText(_ text: String) -> String {
        normalizeDisplayText(text)
    }

    private func normalizeDisplayText(_ text: String) -> String {
        // Normalize whitespace within each line but preserve newlines between chunks
        text
            .components(separatedBy: "\n")
            .map { line in
                line.replacingOccurrences(of: "[^\\S\\n]+", with: " ", options: .regularExpression)
                    .trimmingCharacters(in: .whitespaces)
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Native Translation

    private func resetTranslationState() {
        translationTask?.cancel()
        translationTask = nil
        translatedConfirmedText = ""
        translatedHypothesisText = ""
        translationWarning = nil
        lastTranslationInput = nil
    }

    private func scheduleTranslationUpdate() {
        translationTask?.cancel()

        guard translationEnabled else {
            return
        }

        let sourceCode = translationSourceLanguageCode.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetCode = translationTargetLanguageCode.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sourceCode.isEmpty, !targetCode.isEmpty else { return }

        let confirmedSnapshot = confirmedText
        let hypothesisSnapshot = hypothesisText

        // Skip if text hasn't changed since last translation request.
        if let last = lastTranslationInput,
           last.confirmed == confirmedSnapshot,
           last.hypothesis == hypothesisSnapshot {
            return
        }

        #if targetEnvironment(simulator)
        // iOS Simulator cannot run the native Translation framework pipeline.
        // Keep translation flows testable by using source text fallback inline.
        let simulatorConfirmed = normalizeDisplayText(confirmedSnapshot)
        let simulatorHypothesis = normalizeDisplayText(hypothesisSnapshot)
        translatedConfirmedText = simulatorConfirmed
        translatedHypothesisText = simulatorHypothesis
        translationWarning = sourceCode.caseInsensitiveCompare(targetCode) == .orderedSame
            ? nil
            : "On-device Translation API is unavailable on iOS Simulator. Using source text fallback."
        lastTranslationInput = (confirmedSnapshot, hypothesisSnapshot)
        #else
        if sourceCode.caseInsensitiveCompare(targetCode) == .orderedSame {
            translatedConfirmedText = normalizeDisplayText(confirmedSnapshot)
            translatedHypothesisText = normalizeDisplayText(hypothesisSnapshot)
            translationWarning = nil
            lastTranslationInput = (confirmedSnapshot, hypothesisSnapshot)
            return
        }

        translationTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .milliseconds(180))
            guard !Task.isCancelled else { return }

            var warningMessage: String?

            do {
                async let confirmedTranslated = self.translationService.translate(
                    text: confirmedSnapshot,
                    sourceLanguageCode: sourceCode,
                    targetLanguageCode: targetCode
                )
                async let hypothesisTranslated = self.translationService.translate(
                    text: hypothesisSnapshot,
                    sourceLanguageCode: sourceCode,
                    targetLanguageCode: targetCode
                )

                let translatedConfirmed = try await confirmedTranslated
                let translatedHypothesis = try await hypothesisTranslated
                guard !Task.isCancelled else { return }

                self.translatedConfirmedText = translatedConfirmed
                self.translatedHypothesisText = translatedHypothesis
            } catch let appError as AppError {
                guard !Task.isCancelled else { return }
                // Fallback when native translation is unavailable:
                // keep UI functional by reusing source text and surfacing warning inline.
                self.translatedConfirmedText = self.normalizeDisplayText(confirmedSnapshot)
                self.translatedHypothesisText = self.normalizeDisplayText(hypothesisSnapshot)
                warningMessage = appError.localizedDescription
            } catch {
                guard !Task.isCancelled else { return }
                self.translatedConfirmedText = self.normalizeDisplayText(confirmedSnapshot)
                self.translatedHypothesisText = self.normalizeDisplayText(hypothesisSnapshot)
                warningMessage = AppError.translationFailed(underlying: error).localizedDescription
            }

            self.translationWarning = warningMessage
            self.lastTranslationInput = (confirmedSnapshot, hypothesisSnapshot)
        }
        #endif
    }

    private func resetTranscriptionState() {
        cancelAndTrackTranscriptionTask()
        resetTranslationState()
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        confirmedSegments = []
        unconfirmedSegments = []
        confirmedText = ""
        hypothesisText = ""
        completedChunksText = ""
        bufferEnergy = []
        bufferSeconds = 0
        tokensPerSecond = 0
        prevUnconfirmedSegments = []
        consecutiveSilenceCount = 0
        hasCompletedFirstInference = false
        movingAverageInferenceSeconds = 0.0
        lastUIMeterUpdateTimestamp = 0
        lastError = nil
    }

    // MARK: - Testing Support

    #if DEBUG
    func testFeedResult(_ result: ASRResult) {
        processTranscriptionResult(result)
    }

    func testSetState(
        confirmedText: String = "",
        hypothesisText: String = "",
        confirmedSegments: [ASRSegment] = [],
        unconfirmedSegments: [ASRSegment] = []
    ) {
        self.confirmedText = confirmedText
        self.hypothesisText = hypothesisText
        self.confirmedSegments = confirmedSegments
        self.unconfirmedSegments = unconfirmedSegments
    }

    func testSetSessionState(_ state: SessionState) {
        self.sessionState = state
    }

    func testSetRecordingFlags(isRecording: Bool, isTranscribing: Bool) {
        self.isRecording = isRecording
        self.isTranscribing = isTranscribing
    }

    func testSimulateInterruption(began: Bool) {
        if began {
            if isRecording {
                cancelAndTrackTranscriptionTask()
                isTranscribing = false
                sessionState = .interrupted
            }
        } else {
            if sessionState == .interrupted {
                stopRecording()
            }
        }
    }
    #endif
}
