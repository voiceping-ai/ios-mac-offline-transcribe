import Foundation

struct ModelInfo: Identifiable, Hashable {
    let id: String
    let displayName: String
    let parameterCount: String
    let sizeOnDisk: String
    let description: String
    let family: ModelFamily
    let engineType: ASREngineType
    let languages: String

    /// WhisperKit variant name (e.g. "openai_whisper-tiny"). Only for WhisperKit models.
    let variant: String?

    /// sherpa-onnx model config. Only for sherpa-onnx models.
    let sherpaModelConfig: SherpaModelConfig?
    /// Qwen ASR model config. Only for Qwen ASR models.
    let qwenModelConfig: QwenModelConfig?
    /// Logical card ID used by the new backend-aware catalog.
    let cardId: String?
    /// Runtime variant ID from the backend-aware catalog.
    let runtimeVariantId: String?
    /// Requested backend from the backend-aware catalog.
    let backend: InferenceBackend?

    init(
        id: String,
        displayName: String,
        parameterCount: String,
        sizeOnDisk: String,
        description: String,
        family: ModelFamily,
        engineType: ASREngineType,
        languages: String,
        variant: String?,
        sherpaModelConfig: SherpaModelConfig?,
        qwenModelConfig: QwenModelConfig? = nil,
        cardId: String? = nil,
        runtimeVariantId: String? = nil,
        backend: InferenceBackend? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.parameterCount = parameterCount
        self.sizeOnDisk = sizeOnDisk
        self.description = description
        self.family = family
        self.engineType = engineType
        self.languages = languages
        self.variant = variant
        self.sherpaModelConfig = sherpaModelConfig
        self.qwenModelConfig = qwenModelConfig
        self.cardId = cardId
        self.runtimeVariantId = runtimeVariantId
        self.backend = backend
    }

    static let availableModels: [ModelInfo] = [
        // MARK: - Whisper (WhisperKit)
        ModelInfo(
            id: "whisper-tiny",
            displayName: "Whisper Tiny",
            parameterCount: "39M",
            sizeOnDisk: "~80 MB",
            description: "Fastest, lower accuracy. Good for quick notes.",
            family: .whisper,
            engineType: .whisperKit,
            languages: "99 languages",
            variant: "openai_whisper-tiny",
            sherpaModelConfig: nil
        ),
        ModelInfo(
            id: "whisper-base",
            displayName: "Whisper Base",
            parameterCount: "74M",
            sizeOnDisk: "~150 MB",
            description: "Balanced speed and accuracy. Recommended.",
            family: .whisper,
            engineType: .whisperKit,
            languages: "English",
            variant: "openai_whisper-base.en",
            sherpaModelConfig: nil
        ),
        ModelInfo(
            id: "whisper-small",
            displayName: "Whisper Small",
            parameterCount: "244M",
            sizeOnDisk: "~500 MB",
            description: "Higher accuracy, slower. Best for important recordings.",
            family: .whisper,
            engineType: .whisperKit,
            languages: "99 languages",
            variant: "openai_whisper-small",
            sherpaModelConfig: nil
        ),
        ModelInfo(
            id: "whisper-large-v3-turbo",
            displayName: "Whisper Large V3 Turbo",
            parameterCount: "809M",
            sizeOnDisk: "~600 MB",
            description: "Near-SOTA accuracy (7.75% WER). Best quality, larger download.",
            family: .whisper,
            engineType: .whisperKit,
            languages: "99 languages",
            variant: "openai_whisper-large-v3_turbo",
            sherpaModelConfig: nil
        ),
        ModelInfo(
            id: "whisper-large-v3-turbo-compressed",
            displayName: "Whisper Large V3 Turbo (Compressed)",
            parameterCount: "809M",
            sizeOnDisk: "~1 GB",
            description: "Compressed Turbo. Slightly lower quality, smaller download.",
            family: .whisper,
            engineType: .whisperKit,
            languages: "99 languages",
            // Compatibility fallback: route compressed ID to stable turbo runtime.
            variant: "openai_whisper-large-v3_turbo",
            sherpaModelConfig: nil
        ),

        // MARK: - Moonshine (sherpa-onnx offline)
        ModelInfo(
            id: "moonshine-tiny",
            displayName: "Moonshine Tiny",
            parameterCount: "27M",
            sizeOnDisk: "~125 MB",
            description: "Ultra-fast English. Beats Whisper Tiny at 30% fewer params.",
            family: .moonshine,
            engineType: .sherpaOnnxOffline,
            languages: "English",
            variant: nil,
            sherpaModelConfig: SherpaModelConfig(
                repoName: "sherpa-onnx-moonshine-tiny-en-int8",
                tokens: "tokens.txt",
                modelType: .moonshine,
                preprocessor: "preprocess.onnx",
                encoder: "encode.int8.onnx",
                uncachedDecoder: "uncached_decode.int8.onnx",
                cachedDecoder: "cached_decode.int8.onnx"
            )
        ),
        ModelInfo(
            id: "moonshine-base",
            displayName: "Moonshine Base",
            parameterCount: "61M",
            sizeOnDisk: "~280 MB",
            description: "Fast English with higher accuracy than Tiny.",
            family: .moonshine,
            engineType: .sherpaOnnxOffline,
            languages: "English",
            variant: nil,
            sherpaModelConfig: SherpaModelConfig(
                repoName: "sherpa-onnx-moonshine-base-en-int8",
                tokens: "tokens.txt",
                modelType: .moonshine,
                preprocessor: "preprocess.onnx",
                encoder: "encode.int8.onnx",
                uncachedDecoder: "uncached_decode.int8.onnx",
                cachedDecoder: "cached_decode.int8.onnx"
            )
        ),

        // MARK: - SenseVoice (sherpa-onnx offline)
        ModelInfo(
            id: "sensevoice-small",
            displayName: "SenseVoice Small",
            parameterCount: "234M",
            sizeOnDisk: "~240 MB",
            description: "5x faster than Whisper. Chinese, English, Japanese, Korean, Cantonese.",
            family: .senseVoice,
            engineType: .sherpaOnnxOffline,
            languages: "zh/en/ja/ko/yue",
            variant: nil,
            sherpaModelConfig: SherpaModelConfig(
                repoName: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                tokens: "tokens.txt",
                modelType: .senseVoice,
                senseVoiceModel: "model.int8.onnx"
            )
        ),

        // MARK: - Zipformer (sherpa-onnx streaming)
        ModelInfo(
            id: "zipformer-20m",
            displayName: "Zipformer Streaming",
            parameterCount: "20M",
            sizeOnDisk: "~46 MB",
            description: "Real-time streaming English. Ultra-low latency.",
            family: .zipformer,
            engineType: .sherpaOnnxStreaming,
            languages: "English",
            variant: nil,
            sherpaModelConfig: SherpaModelConfig(
                repoName: "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
                tokens: "tokens.txt",
                modelType: .zipformerTransducer,
                encoder: "encoder-epoch-99-avg-1.int8.onnx",
                uncachedDecoder: "decoder-epoch-99-avg-1.onnx",
                joiner: "joiner-epoch-99-avg-1.int8.onnx"
            )
        ),

        // MARK: - Omnilingual (sherpa-onnx offline CTC)
        ModelInfo(
            id: "omnilingual-300m",
            displayName: "Omnilingual 300M",
            parameterCount: "300M",
            sizeOnDisk: "~365 MB",
            description: "1,600+ languages. Facebook MMS CTC model, int8 quantized.",
            family: .omnilingual,
            engineType: .sherpaOnnxOffline,
            languages: "1,600+ languages",
            variant: nil,
            sherpaModelConfig: SherpaModelConfig(
                repoName: "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
                tokens: "tokens.txt",
                modelType: .omnilingualCtc,
                omnilingualModel: "model.int8.onnx"
            )
        ),

        // MARK: - Parakeet (FluidAudio, CoreML)
        ModelInfo(
            id: "parakeet-tdt-v3",
            displayName: "Parakeet TDT 0.6B",
            parameterCount: "600M",
            sizeOnDisk: "~600 MB",
            description: "Best English WER (2.5%). 25 European languages. ~1.2 GB RAM.",
            family: .parakeet,
            engineType: .fluidAudio,
            languages: "25 European languages",
            variant: nil,
            sherpaModelConfig: nil
        ),

        // MARK: - Apple Speech (built-in)
        ModelInfo(
            id: "apple-speech",
            displayName: "Apple Speech",
            parameterCount: "System",
            sizeOnDisk: "Built-in",
            description: "Apple's native on-device speech recognition. 50+ languages, no download required.",
            family: .appleSpeech,
            engineType: .appleSpeech,
            languages: "50+ languages",
            variant: nil,
            sherpaModelConfig: nil
        ),

        // MARK: - Qwen ASR (pure C inference)
        ModelInfo(
            id: "qwen3-asr-0.6b",
            displayName: "Qwen3 ASR 0.6B",
            parameterCount: "600M",
            sizeOnDisk: "~1.8 GB",
            description: "30 languages. Pure C inference with NEON + Accelerate. ~2.7 GB RAM.",
            family: .qwenASR,
            engineType: .qwenASR,
            languages: "30 languages",
            variant: nil,
            sherpaModelConfig: nil,
            qwenModelConfig: QwenModelConfig(
                repoId: "Qwen/Qwen3-ASR-0.6B",
                files: ["config.json", "generation_config.json", "model.safetensors", "vocab.json", "merges.txt"]
            )
        ),

        // MARK: - Qwen ASR (MLX, macOS only)
        ModelInfo(
            id: "qwen3-asr-0.6b-mlx",
            displayName: "Qwen3 ASR 0.6B (MLX)",
            parameterCount: "600M",
            sizeOnDisk: "~1 GB (8-bit)",
            description: "30 languages. MLX Metal GPU acceleration on Apple Silicon. 8-bit quantized.",
            family: .qwenASR,
            engineType: .mlx,
            languages: "30 languages",
            variant: nil,
            sherpaModelConfig: nil
        ),

        // MARK: - Qwen ASR (ONNX Runtime, INT8)
        ModelInfo(
            id: "qwen3-asr-0.6b-onnx",
            displayName: "Qwen3 ASR 0.6B (ONNX)",
            parameterCount: "600M",
            sizeOnDisk: "~1.6 GB (INT8)",
            description: "30 languages. ONNX Runtime INT8 quantized. Cross-platform.",
            family: .qwenASR,
            engineType: .qwenOnnx,
            languages: "30 languages",
            variant: nil,
            sherpaModelConfig: nil,
            qwenModelConfig: QwenModelConfig(
                repoId: "jima/qwen3-asr-0.6b-onnx-int8",
                files: [
                    "encoder.int8.onnx",
                    "decoder_prefill.int8.onnx",
                    "decoder_decode.int8.onnx",
                    "embed_tokens.fp16.npy",
                    "vocab.json",
                    "config.json",
                    "tokens.json"
                ]
            )
        ),
    ]

    static let defaultModel = availableModels.first { $0.id == "whisper-base" }!
    private static let familyDisplayOrder: [ModelFamily] = [
        .senseVoice,
        .whisper,
        .moonshine,
        .zipformer,
        .omnilingual,
        .parakeet,
        .qwenASR,
        .appleSpeech
    ]
    private static let cachedSupportedModels: [ModelInfo] = {
        var models = availableModels
        if !FluidAudioEngine.isDeviceSupported {
            models = models.filter { $0.engineType != .fluidAudio }
        }
        if !BackendCapabilities.isBackendSupported(.mlx) {
            models = models.filter { $0.engineType != .mlx }
        }
        return models
    }()
    private static let cachedModelsByFamily: [(family: ModelFamily, models: [ModelInfo])] = {
        let grouped = Dictionary(grouping: cachedSupportedModels, by: \.family)
        return familyDisplayOrder.compactMap { family in
            guard let models = grouped[family], !models.isEmpty else { return nil }
            return (family: family, models: models)
        }
    }()

    static var legacyModelCards: [ModelCard] {
        supportedModels.map { model in
            let legacyVariant = ModelRuntimeVariant(
                id: "\(model.id)-legacy",
                backend: .legacy,
                engineType: model.engineType,
                runtimeLabel: model.inferenceMethodLabel,
                platforms: RuntimePlatform.allCases,
                architectures: [],
                minimumOSVersion: nil,
                legacyModelId: model.id,
                artifacts: [],
                isEnabled: true
            )
            return ModelCard(
                id: model.id,
                displayName: model.displayName,
                parameterCount: model.parameterCount,
                sizeOnDisk: model.sizeOnDisk,
                description: model.description,
                family: model.family,
                languages: model.languages,
                runtimeVariants: [legacyVariant]
            )
        }
    }

    static func from(card: ModelCard, variant: ModelRuntimeVariant) -> ModelInfo {
        if let legacyModelId = variant.legacyModelId,
           let legacyModel = availableModels.first(where: { $0.id == legacyModelId }) {
            return legacyModel.withCatalogContext(
                cardId: card.id,
                variantId: variant.id,
                backend: variant.backend
            )
        }

        return ModelInfo(
            id: "\(card.id)-\(variant.backend.rawValue)",
            displayName: card.displayName,
            parameterCount: card.parameterCount,
            sizeOnDisk: card.sizeOnDisk,
            description: card.description,
            family: card.family,
            engineType: variant.engineType,
            languages: card.languages,
            variant: nil,
            sherpaModelConfig: nil,
            cardId: card.id,
            runtimeVariantId: variant.id,
            backend: variant.backend
        )
    }

    var inferenceMethodLabel: String {
        switch engineType {
        case .whisperKit:
            return "WhisperKit · CoreML"
        case .sherpaOnnxOffline:
            return "sherpa-onnx · ONNX Runtime"
        case .sherpaOnnxStreaming:
            return "sherpa-onnx · Streaming"
        case .fluidAudio:
            return "FluidAudio · CoreML"
        case .cactus:
            return "Cactus · Optimized Runtime"
        case .mlx:
            return "MLX · Metal GPU"
        case .appleSpeech:
            return "Apple Speech · On-device"
        case .qwenASR:
            return "QwenASR · Pure C"
        case .qwenOnnx:
            return "QwenASR · ONNX Runtime"
        }
    }

    /// Backward-compat: find a model by old-style ID ("tiny" → "whisper-tiny").
    static func findByLegacyId(_ legacyId: String) -> ModelInfo? {
        if let model = availableModels.first(where: { $0.id == legacyId }) {
            return model
        }
        return availableModels.first(where: { $0.id == "whisper-\(legacyId)" })
    }

    /// Models filtered by device capability.
    static var supportedModels: [ModelInfo] {
        cachedSupportedModels
    }

    /// Models grouped by family for UI display.
    static var modelsByFamily: [(family: ModelFamily, models: [ModelInfo])] {
        cachedModelsByFamily
    }

    func withCatalogContext(
        cardId: String?,
        variantId: String?,
        backend: InferenceBackend?
    ) -> ModelInfo {
        ModelInfo(
            id: id,
            displayName: displayName,
            parameterCount: parameterCount,
            sizeOnDisk: sizeOnDisk,
            description: description,
            family: family,
            engineType: engineType,
            languages: languages,
            variant: variant,
            sherpaModelConfig: sherpaModelConfig,
            qwenModelConfig: qwenModelConfig,
            cardId: cardId,
            runtimeVariantId: variantId,
            backend: backend
        )
    }
}

// MARK: - sherpa-onnx Model Config

enum SherpaModelType: String, Codable, Sendable {
    case moonshine
    case senseVoice
    case zipformerTransducer
    case omnilingualCtc
}

struct SherpaModelConfig: Hashable, Sendable {
    let repoName: String
    let tokens: String
    let modelType: SherpaModelType

    // Moonshine offline
    var preprocessor: String?
    var encoder: String?
    var uncachedDecoder: String?
    var cachedDecoder: String?

    // SenseVoice
    var senseVoiceModel: String?

    // Zipformer streaming transducer
    var joiner: String?

    // Omnilingual CTC
    var omnilingualModel: String?

    /// All files needed for this model (used for individual file downloads).
    var allFiles: [String] {
        var files = [tokens]
        switch modelType {
        case .moonshine:
            if let p = preprocessor { files.append(p) }
            if let e = encoder { files.append(e) }
            if let u = uncachedDecoder { files.append(u) }
            if let c = cachedDecoder { files.append(c) }
        case .senseVoice:
            if let m = senseVoiceModel { files.append(m) }
        case .zipformerTransducer:
            if let e = encoder { files.append(e) }
            if let d = uncachedDecoder { files.append(d) }
            if let j = joiner { files.append(j) }
        case .omnilingualCtc:
            if let m = omnilingualModel { files.append(m) }
        }
        return files
    }
}

// MARK: - Qwen ASR Model Config

struct QwenModelConfig: Hashable, Sendable {
    /// HuggingFace repo ID (e.g. "Qwen/Qwen3-ASR-0.6B").
    let repoId: String
    /// Files to download from the repo.
    let files: [String]

    /// Local directory name derived from the repo ID.
    var localDirName: String {
        repoId.replacingOccurrences(of: "/", with: "_")
    }
}
