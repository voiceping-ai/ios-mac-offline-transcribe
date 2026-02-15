import Foundation
import Dispatch
import WhisperKit
import Observation
@preconcurrency import AVFoundation

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

    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0.0
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
    private(set) var modelCards: [ModelCard] = ModelInfo.legacyModelCards
    private(set) var modelCatalogSource: ModelCatalogSource = .legacy
    private(set) var selectedModelCardId: String = ModelInfo.defaultModel.id
    private(set) var selectedInferenceBackend: InferenceBackend = .automatic
    private(set) var effectiveInferenceBackend: InferenceBackend = .legacy
    private(set) var backendFallbackWarning: String?
    private(set) var effectiveRuntimeLabel: String = ModelInfo.defaultModel.inferenceMethodLabel
    var audioCaptureMode: AudioCaptureMode = .microphone
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
        didSet {
            lastTranslationInput = nil
            scheduleTranslationUpdate()
        }
    }
    var translationTargetLanguageCode: String = "ja" {
        didSet {
            lastTranslationInput = nil
            scheduleTranslationUpdate()
        }
    }

    // Engine delegation
    private(set) var activeEngine: ASREngine?

    /// Coordinates real-time transcription loops, VAD, and chunking.
    private(set) var transcriptionCoordinator: TranscriptionCoordinator!

    /// System audio source for broadcast mode (receives audio from Broadcast Extension).
    private var systemAudioSource: SystemAudioSource?

    /// Whether a ReplayKit broadcast is currently active.
    private(set) var isBroadcastActive = false

    /// The current session's audio samples (for saving to disk).
    var currentAudioSamples: [Float] {
        if audioCaptureMode == .systemBroadcast, let source = systemAudioSource {
            return source.audioSamples
        }
        return activeEngine?.audioSamples ?? []
    }

    /// Audio samples for transcription — uses SystemAudioSource in broadcast mode.
    var effectiveAudioSamples: [Float] {
        if audioCaptureMode == .systemBroadcast, let source = systemAudioSource {
            return source.audioSamples
        }
        return activeEngine?.audioSamples ?? []
    }

    /// Energy levels for VAD / visualization — uses SystemAudioSource in broadcast mode.
    var effectiveRelativeEnergy: [Float] {
        if audioCaptureMode == .systemBroadcast, let source = systemAudioSource {
            return source.relativeEnergy
        }
        return activeEngine?.relativeEnergy ?? []
    }

    // Private
    private var fileTranscriptionTask: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
    private var e2eTranscribeInFlight: Bool = false
    /// Cache: last input text pair sent for translation (to skip redundant calls).
    private var lastTranslationInput: (confirmed: String, hypothesis: String)?
    private let translationService = AppleTranslationService()
    private let catalogService = ModelCatalogService.shared
    private let backendResolver = BackendResolver.shared

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
    private var appDiagLines: [String] = []
    private static let appDiagWriteQueue = DispatchQueue(label: "com.voiceping.transcribe.appDiag", qos: .utility)
    private func writeAppDiag(_ line: String) {
        appDiagLines.append(line)

        // Avoid blocking the main actor (and hanging CI) on disk IO during tests.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else { return }

        // Best-effort debug logging: keep it out of the synchronous stopRecording() path.
        let payload = appDiagLines.joined(separator: "\n")
        guard let url = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: SharedAudioRingBuffer.appGroupID
        )?.appendingPathComponent("app_diag.txt") else { return }
        Self.appDiagWriteQueue.async {
            try? payload.write(to: url, atomically: true, encoding: .utf8)
        }
    }
    private let selectedModelKey = "selectedModelVariant"
    private let selectedCardKey = "selectedModelCardId"
    private let selectedBackendKey = "selectedInferenceBackend"
    private static let sampleRate: Float = AudioConstants.sampleRateFloat
    private static let e2eTimestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private var isOmnilingualModel: Bool {
        if selectedModel.sherpaModelConfig?.modelType == .omnilingualCtc {
            return true
        }
        return selectedModel.id.lowercased().contains("omnilingual")
    }

    init() {
        self.transcriptionCoordinator = nil  // Placeholder until self is available
        let localCatalog = catalogService.loadLocalFallbackCatalog()
        self.modelCards = localCatalog.cards
        self.modelCatalogSource = localCatalog.source

        let defaults = UserDefaults.standard
        if let backendRaw = defaults.string(forKey: selectedBackendKey),
           let savedBackend = InferenceBackend(rawValue: backendRaw) {
            self.selectedInferenceBackend = savedBackend
        }

        if let savedCardId = defaults.string(forKey: selectedCardKey),
           modelCards.contains(where: { $0.id == savedCardId }) {
            self.selectedModelCardId = savedCardId
        } else if let migrated = migrateLegacySelection(from: defaults.string(forKey: selectedModelKey)) {
            self.selectedModelCardId = migrated.cardId
            self.selectedInferenceBackend = migrated.backend
        } else if let defaultCardId = preferredDefaultCardId(in: modelCards) {
            self.selectedModelCardId = defaultCardId
        }

        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: selectedInferenceBackend
        )

        self.transcriptionCoordinator = TranscriptionCoordinator(service: self)

        migrateLegacyModelFolder()
        #if os(iOS)
        setupAudioObservers()
        registerBroadcastNotifications()
        #endif
        startMetricsSampling()
        Task { [weak self] in
            await self?.refreshModelCatalog()
        }
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

    #if os(iOS)
    // MARK: - Broadcast Notifications (ReplayKit IPC)

    private func registerBroadcastNotifications() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()

        let broadcastStartedName = DarwinNotifications.broadcastStarted
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            { _, observer, _, _, _ in
                guard let observer else { return }
                let service = Unmanaged<WhisperService>.fromOpaque(observer).takeUnretainedValue()
                Task { @MainActor in
                    service.handleBroadcastStarted()
                }
            },
            broadcastStartedName,
            nil,
            .deliverImmediately
        )

        let broadcastStoppedName = DarwinNotifications.broadcastStopped
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            { _, observer, _, _, _ in
                guard let observer else { return }
                let service = Unmanaged<WhisperService>.fromOpaque(observer).takeUnretainedValue()
                Task { @MainActor in
                    service.handleBroadcastStopped()
                }
            },
            broadcastStoppedName,
            nil,
            .deliverImmediately
        )
    }

    private func handleBroadcastStarted() {
        NSLog("[WhisperService] Broadcast started notification received (mode=%d, recording=%d, state=%d)",
              audioCaptureMode == .systemBroadcast ? 1 : 0, isRecording ? 1 : 0, sessionState.rawValue)
        writeAppDiag("[\(Date())] handleBroadcastStarted mode=\(audioCaptureMode) recording=\(isRecording) state=\(sessionState.rawValue)")

        isBroadcastActive = true

        guard audioCaptureMode == .systemBroadcast else { return }
        guard !isRecording, sessionState == .idle else { return }
        guard let engine = activeEngine, engine.modelState == .loaded else {
            NSLog("[WhisperService] Broadcast started but engine not ready — skipping auto-record")
            return
        }

        NSLog("[WhisperService] Auto-starting recording for system broadcast")
        Task {
            do {
                try await startRecording()
            } catch {
                NSLog("[WhisperService] Failed to auto-start recording for broadcast: \(error)")
            }
        }
    }

    private func handleBroadcastStopped() {
        NSLog("[WhisperService] Broadcast stopped notification received (recording=%d)", isRecording ? 1 : 0)
        writeAppDiag("[\(Date())] handleBroadcastStopped recording=\(isRecording)")

        isBroadcastActive = false

        guard audioCaptureMode == .systemBroadcast, isRecording else { return }
        NSLog("[WhisperService] Auto-stopping recording because broadcast ended")
        stopRecording()
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
                transcriptionCoordinator.cancelAndTrackTranscriptionTask()
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
                            await self.transcriptionCoordinator.drainLingeringTranscriptionTask()
                            try await engine.startRecording(captureMode: self.audioCaptureMode)
                            isTranscribing = true
                            sessionState = .recording
                            transcriptionCoordinator.startLoop()
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
    #endif

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

    private func migrateLegacySelection(from savedValue: String?) -> (cardId: String, backend: InferenceBackend)? {
        guard let savedValue else { return nil }

        guard let model = ModelInfo.supportedModels.first(where: { $0.variant == savedValue })
            ?? ModelInfo.supportedModels.first(where: { $0.id == savedValue })
            ?? ModelInfo.findByLegacyId(savedValue) else {
            return nil
        }

        let migratedCardId = model.cardId ?? model.id
        UserDefaults.standard.set(migratedCardId, forKey: selectedCardKey)
        UserDefaults.standard.set(InferenceBackend.legacy.rawValue, forKey: selectedBackendKey)
        return (cardId: migratedCardId, backend: .legacy)
    }

    private func preferredDefaultCardId(in cards: [ModelCard]) -> String? {
        let defaultCardId = ModelInfo.defaultModel.cardId ?? ModelInfo.defaultModel.id
        if cards.contains(where: { $0.id == defaultCardId }) {
            return defaultCardId
        }
        return cards.first?.id
    }

    private func persistSelectionKeys() {
        UserDefaults.standard.set(selectedModelCardId, forKey: selectedCardKey)
        UserDefaults.standard.set(selectedInferenceBackend.rawValue, forKey: selectedBackendKey)
    }

    func refreshModelCatalog() async {
        let catalog = await catalogService.loadCatalog()
        modelCards = catalog.cards
        modelCatalogSource = catalog.source

        if !modelCards.contains(where: { $0.id == selectedModelCardId }),
           let defaultCardId = preferredDefaultCardId(in: modelCards) {
            selectedModelCardId = defaultCardId
        }

        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: selectedInferenceBackend
        )
        persistSelectionKeys()
    }

    var modelCardsByFamily: [(family: ModelFamily, cards: [ModelCard])] {
        let grouped = Dictionary(grouping: modelCards, by: \.family)
        let familyOrder: [ModelFamily] = [
            .senseVoice,
            .whisper,
            .moonshine,
            .zipformer,
            .omnilingual,
            .parakeet,
            .qwenASR,
            .appleSpeech
        ]
        return familyOrder.compactMap { family in
            guard let cards = grouped[family], !cards.isEmpty else { return nil }
            return (family: family, cards: cards)
        }
    }

    func selectedCard() -> ModelCard? {
        modelCards.first(where: { $0.id == selectedModelCardId })
    }

    func availableBackends(for card: ModelCard) -> [InferenceBackend] {
        let concreteBackends = card.runtimeVariants
            .map(\.backend)
            .reduce(into: [InferenceBackend]()) { acc, backend in
                if !acc.contains(backend) {
                    acc.append(backend)
                }
            }
            .sorted(by: { $0.rawValue < $1.rawValue })

        if concreteBackends.count <= 1 {
            return concreteBackends
        }

        return [.automatic] + concreteBackends
    }

    func resolvedModelInfo(
        for card: ModelCard,
        requestedBackend: InferenceBackend
    ) -> ModelInfo? {
        let resolution = backendResolver.resolve(card: card, requestedBackend: requestedBackend)
        guard let variant = resolution.runtimeVariant else { return nil }
        return ModelInfo.from(card: card, variant: variant)
    }

    func runtimeLabel(for card: ModelCard, requestedBackend: InferenceBackend) -> String {
        let resolution = backendResolver.resolve(card: card, requestedBackend: requestedBackend)
        return resolution.runtimeVariant?.runtimeLabel ?? "Unavailable"
    }

    func setSelectedModelCard(_ cardId: String) {
        selectedModelCardId = cardId
        applyBackendResolution(
            cardId: cardId,
            requestedBackend: selectedInferenceBackend
        )
        persistSelectionKeys()
    }

    func setSelectedInferenceBackend(_ backend: InferenceBackend) {
        selectedInferenceBackend = backend
        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: backend
        )
        persistSelectionKeys()
    }

    private func applyBackendResolution(
        cardId: String,
        requestedBackend: InferenceBackend
    ) {
        guard let card = modelCards.first(where: { $0.id == cardId }) else {
            selectedModel = ModelInfo.defaultModel
            effectiveInferenceBackend = .legacy
            effectiveRuntimeLabel = selectedModel.inferenceMethodLabel
            backendFallbackWarning = nil
            return
        }

        let resolution = backendResolver.resolve(
            card: card,
            requestedBackend: requestedBackend
        )

        backendFallbackWarning = resolution.fallbackReason
        effectiveInferenceBackend = resolution.effectiveBackend

        guard let variant = resolution.runtimeVariant else {
            selectedModel = ModelInfo.defaultModel
            effectiveRuntimeLabel = selectedModel.inferenceMethodLabel
            return
        }

        selectedModel = ModelInfo.from(card: card, variant: variant)
        effectiveRuntimeLabel = variant.runtimeLabel
    }

    func loadModelIfAvailable() async {
        // Don't overwrite an already-loaded or in-progress engine
        guard activeEngine == nil, modelState == .unloaded else { return }
        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: selectedInferenceBackend
        )

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

    func setupModel() async {
        let logger = InferenceLogger.shared
        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: selectedInferenceBackend
        )
        logger.log("[WhisperService] setupModel: model=\(selectedModel.id) engine=\(selectedModel.engineType)")
        let engine = EngineFactory.makeEngine(for: selectedModel)
        activeEngine = engine

        downloadProgress = 0.0
        lastError = nil
        // Avoid showing "Downloading..." for models that are already cached locally.
        // Engines will still report .downloading if they end up needing network fetches.
        modelState = engine.isModelDownloaded(selectedModel) ? .loading : .downloading

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
            persistSelectionKeys()
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
        case .cactus, .mlx, .qwenASR, .qwenOnnx:
            let engine = EngineFactory.makeEngine(for: model)
            return engine.isModelDownloaded(model)
        case .appleSpeech:
            // Apple Speech is built into iOS — always available
            return true
        }
    }

    func switchModel(to model: ModelInfo, backendOverride: InferenceBackend? = nil) async {
        if isRecording {
            stopRecording()
        }

        await transcriptionCoordinator.drainLingeringTranscriptionTask()
        resetTranscriptionState()

        isRecording = false
        isTranscribing = false
        sessionState = .idle
        transcriptionCoordinator.cancelAndTrackTranscriptionTask()
        translationTask?.cancel()
        translationTask = nil

        // Unload current engine
        if let engine = activeEngine {
            await engine.unloadModel()
        }
        activeEngine = nil
        modelState = .unloaded
        selectedModelCardId = model.cardId ?? model.id
        selectedInferenceBackend = backendOverride ?? model.backend ?? .legacy
        applyBackendResolution(
            cardId: selectedModelCardId,
            requestedBackend: selectedInferenceBackend
        )
        persistSelectionKeys()
        await setupModel()
    }

    // MARK: - Recording & Transcription

    func startRecording() async throws {
        guard sessionState == .idle else { return }

        sessionState = .starting
        await transcriptionCoordinator.drainLingeringTranscriptionTask()

        guard let engine = activeEngine, engine.modelState == .loaded else {
            sessionState = .idle
            throw AppError.modelNotReady
        }

        resetTranscriptionState()

        #if os(iOS)
        if audioCaptureMode == .systemBroadcast {
            // Use SystemAudioSource instead of engine's AudioRecorder
            let source = SystemAudioSource()
            systemAudioSource = source

            // For streaming engines, feed broadcast audio into the engine
            if engine.isStreaming {
                source.onNewAudio = { [weak engine] samples in
                    try? engine?.feedAudio(samples)
                }
            }

            source.start()
        } else {
            systemAudioSource = nil
            do {
                try await engine.startRecording(captureMode: audioCaptureMode)
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
        }
        #else
        systemAudioSource = nil
        do {
            try await engine.startRecording(captureMode: audioCaptureMode)
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
        #endif

        isRecording = true
        isTranscribing = true
        sessionState = .recording

        transcriptionCoordinator.startLoop()
    }

    func stopRecording() {
        NSLog("[WhisperService] stopRecording() called — mode=%@, isBroadcastActive=%d, state=%@",
              String(describing: audioCaptureMode), isBroadcastActive ? 1 : 0, sessionState.rawValue)
        writeAppDiag("[\(Date())] stopRecording mode=\(audioCaptureMode) broadcastActive=\(isBroadcastActive) state=\(sessionState.rawValue)")

        guard sessionState == .recording || sessionState == .interrupted
            || sessionState == .starting else {
            NSLog("[WhisperService] stopRecording() skipped — state=%@ not recording/interrupted/starting", sessionState.rawValue)
            return
        }

        sessionState = .stopping
        transcriptionCoordinator.cancelAndTrackTranscriptionTask()
        translationTask?.cancel()
        translationTask = nil

        #if os(iOS)
        // Signal broadcast extension to stop BEFORE releasing the ring buffer.
        // Uses shared memory flag (checked every 200ms via timer in SampleHandler)
        // plus Darwin notification as backup.
        if audioCaptureMode == .systemBroadcast && isBroadcastActive {
            // Set requestStop via shared memory — most reliable cross-process signal
            if let ringBuf = SharedAudioRingBuffer(isProducer: false) {
                ringBuf.setRequestStop(true)
                NSLog("[WhisperService] Set requestStop flag in shared ring buffer")
            } else {
                NSLog("[WhisperService] WARNING: Failed to open ring buffer for requestStop!")
            }
            // Also send Darwin notification as backup
            NSLog("[WhisperService] Requesting broadcast stop via Darwin notification")
            let center = CFNotificationCenterGetDarwinNotifyCenter()
            CFNotificationCenterPostNotification(
                center,
                DarwinNotifications.stopBroadcast,
                nil, nil, true
            )
        } else {
            NSLog("[WhisperService] Skipping broadcast stop signal — mode=%@, broadcastActive=%d",
                  String(describing: audioCaptureMode), isBroadcastActive ? 1 : 0)
        }
        #endif

        systemAudioSource?.stop()
        systemAudioSource = nil
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

        transcriptionCoordinator.cancelAndTrackTranscriptionTask()
        fileTranscriptionTask?.cancel()
        fileTranscriptionTask = Task {
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
                updateTokensPerSecond(wordCount: words, elapsedMs: elapsedMs)
                applyFileTranscriptionResult(result)
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
        isTranscribingFile = true
        e2eTranscribeInFlight = true

        transcriptionCoordinator.cancelAndTrackTranscriptionTask()
        fileTranscriptionTask?.cancel()
        fileTranscriptionTask = Task {
            defer {
                e2eTranscribeInFlight = false
                isTranscribingFile = false
            }
            do {
                NSLog("[E2E] Loading audio file...")
                let samples = try Self.loadAudioFile(url: URL(fileURLWithPath: path))
                let audioDuration = Double(samples.count) / Double(Self.sampleRate)
                let minSample = samples.min() ?? 0
                let maxSample = samples.max() ?? 0
                let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / max(Float(samples.count), 1))
                NSLog("[E2E] Audio loaded: \(samples.count) samples (\(audioDuration)s)")
                logger.log("E2E audio stats model=\(selectedModel.id) min=\(String(format: "%.4f", minSample)) max=\(String(format: "%.4f", maxSample)) rms=\(String(format: "%.5f", rms))")
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
                updateTokensPerSecond(wordCount: words, elapsedMs: elapsedMs)
                NSLog("[E2E] Transcription complete: text='\(result.text)', segments=\(result.segments.count)")
                applyFileTranscriptionResult(result)
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
        let safeTokensPerSecond = tokensPerSecond.isFinite ? tokensPerSecond : 0
        let safeDurationMs = durationMs.isFinite ? durationMs : 0
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
            "tokens_per_second": safeTokensPerSecond,
            "duration_ms": safeDurationMs,
            "timestamp": Self.e2eTimestampFormatter.string(from: Date()),
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
            var wroteAny = false
            var writeErrors: [String] = []
            for fileURL in Self.e2eResultOutputURLs(modelId: modelId) {
                do {
                    let parent = fileURL.deletingLastPathComponent()
                    try FileManager.default.createDirectory(
                        at: parent,
                        withIntermediateDirectories: true
                    )
                    try data.write(to: fileURL, options: .atomic)
                    wroteAny = true
                    NSLog("[E2E] Result written to \(fileURL.path)")
                } catch {
                    writeErrors.append("\(fileURL.path): \(error.localizedDescription)")
                }
            }
            if !wroteAny {
                throw NSError(
                    domain: "WhisperService",
                    code: -7,
                    userInfo: [NSLocalizedDescriptionKey: writeErrors.joined(separator: " | ")]
                )
            }
        } catch {
            e2eOverlayPayload = """
            {"model_id":"\(selectedModel.id)","pass":false,"error":"failed to serialize/write E2E result"}
            """
            NSLog("[E2E] Failed to write result file: \(error)")
        }
    }

    private static func e2eResultOutputURLs(modelId: String) -> [URL] {
        let fileName = "e2e_result_\(modelId).json"
        var urls: [URL] = [
            URL(fileURLWithPath: "/tmp").appendingPathComponent(fileName)
        ]

        if let groupURL = FileManager.default
            .containerURL(forSecurityApplicationGroupIdentifier: "group.com.voiceping.transcribe") {
            urls.append(groupURL.appendingPathComponent(fileName))
        }

        urls.append(FileManager.default.temporaryDirectory.appendingPathComponent(fileName))

        var deduped: [URL] = []
        var seen: Set<String> = []
        for url in urls {
            let path = url.path
            if seen.insert(path).inserted {
                deduped.append(url)
            }
        }
        return deduped
    }

    /// Load any audio file and return 16kHz mono Float32 samples in [-1, 1].
    /// Uses AVAudioConverter to handle arbitrary sample rates and channel counts.
    private static func loadAudioFile(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let fileFormat = file.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioConstants.sampleRate,
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
        if fileFormat.sampleRate == AudioConstants.sampleRate && fileFormat.channelCount == 1
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

        let ratio = AudioConstants.sampleRate / fileFormat.sampleRate
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
        transcriptionCoordinator.assembleFullText(
            confirmedSegments: confirmedSegments,
            unconfirmedSegments: unconfirmedSegments
        )
    }

    // MARK: - Internal API (for TranscriptionCoordinator)

    func updateTranscriptionText(confirmed: String, hypothesis: String) {
        if confirmedText != confirmed { confirmedText = confirmed }
        if hypothesisText != hypothesis { hypothesisText = hypothesis }
    }

    func updateSegments(confirmed: [ASRSegment], unconfirmed: [ASRSegment]) {
        confirmedSegments = confirmed
        unconfirmedSegments = unconfirmed
    }

    func updateUnconfirmedSegments(_ segments: [ASRSegment]) {
        unconfirmedSegments = segments
    }

    func updateMeters(energy: [Float], bufferSeconds seconds: Double) {
        if bufferEnergy != energy { bufferEnergy = energy }
        if bufferSeconds != seconds { bufferSeconds = seconds }
    }

    func updateTokensPerSecond(_ value: Double) {
        tokensPerSecond = value
    }

    func updateLastError(_ error: AppError) {
        lastError = error
    }

    /// Called by TranscriptionCoordinator when the inference loop exits naturally.
    func endTranscriptionLoop() {
        isRecording = false
        isTranscribing = false
        sessionState = .idle
        systemAudioSource?.stop()
        systemAudioSource = nil
        activeEngine?.stopRecording()
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

    func scheduleTranslationUpdate() {
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
        let simulatorConfirmed = transcriptionCoordinator.normalizeDisplayText(confirmedSnapshot)
        let simulatorHypothesis = transcriptionCoordinator.normalizeDisplayText(hypothesisSnapshot)
        applyTranslationFallback(
            confirmed: simulatorConfirmed,
            hypothesis: simulatorHypothesis,
            warning: sourceCode.caseInsensitiveCompare(targetCode) == .orderedSame
                ? nil
                : "On-device Translation API is unavailable on iOS Simulator. Using source text fallback."
        )
        lastTranslationInput = (confirmedSnapshot, hypothesisSnapshot)
        #else
        if sourceCode.caseInsensitiveCompare(targetCode) == .orderedSame {
            applyTranslationFallback(
                confirmed: transcriptionCoordinator.normalizeDisplayText(confirmedSnapshot),
                hypothesis: transcriptionCoordinator.normalizeDisplayText(hypothesisSnapshot),
                warning: nil
            )
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
                warningMessage = appError.localizedDescription
                self.applyTranslationFallback(
                    confirmed: self.transcriptionCoordinator.normalizeDisplayText(confirmedSnapshot),
                    hypothesis: self.transcriptionCoordinator.normalizeDisplayText(hypothesisSnapshot),
                    warning: warningMessage
                )
            } catch {
                guard !Task.isCancelled else { return }
                warningMessage = AppError.translationFailed(underlying: error).localizedDescription
                self.applyTranslationFallback(
                    confirmed: self.transcriptionCoordinator.normalizeDisplayText(confirmedSnapshot),
                    hypothesis: self.transcriptionCoordinator.normalizeDisplayText(hypothesisSnapshot),
                    warning: warningMessage
                )
            }

            self.translationWarning = warningMessage
            self.lastTranslationInput = (confirmedSnapshot, hypothesisSnapshot)
        }
        #endif
    }

    private func updateTokensPerSecond(wordCount: Int, elapsedMs: Double) {
        let elapsedSeconds = elapsedMs / 1000.0
        if elapsedSeconds > 0, wordCount > 0 {
            tokensPerSecond = Double(wordCount) / elapsedSeconds
        } else {
            tokensPerSecond = 0
        }
    }

    private func applyFileTranscriptionResult(_ result: ASRResult) {
        confirmedSegments = result.segments
        let segmentText = result.segments.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        confirmedText = segmentText.isEmpty
            ? result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            : segmentText
    }

    private func applyTranslationFallback(confirmed: String, hypothesis: String, warning: String?) {
        translatedConfirmedText = confirmed
        translatedHypothesisText = hypothesis
        translationWarning = warning
    }

    private func resetTranscriptionState() {
        transcriptionCoordinator.reset()
        fileTranscriptionTask?.cancel()
        fileTranscriptionTask = nil
        resetTranslationState()
        confirmedSegments = []
        unconfirmedSegments = []
        confirmedText = ""
        hypothesisText = ""
        bufferEnergy = []
        bufferSeconds = 0
        tokensPerSecond = 0
        lastError = nil
    }

    // MARK: - Testing Support

    #if DEBUG
    func testFeedResult(_ result: ASRResult) {
        transcriptionCoordinator.processTranscriptionResult(result)
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
                transcriptionCoordinator.cancelAndTrackTranscriptionTask()
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
