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
            languages: "99 languages",
            variant: "openai_whisper-base",
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
    ]

    static let defaultModel = availableModels.first { $0.id == "whisper-base" }!

    var inferenceMethodLabel: String {
        switch engineType {
        case .whisperKit:
            return "CoreML (WhisperKit)"
        case .sherpaOnnxOffline:
            return "sherpa-onnx offline (ONNX Runtime)"
        case .sherpaOnnxStreaming:
            return "sherpa-onnx streaming (ONNX Runtime)"
        case .fluidAudio:
            return "CoreML (FluidAudio)"
        }
    }

    /// Backward-compat: find a model by old-style ID ("tiny" → "whisper-tiny").
    static func findByLegacyId(_ legacyId: String) -> ModelInfo? {
        if let model = availableModels.first(where: { $0.id == legacyId }) {
            return model
        }
        return availableModels.first(where: { $0.id == "whisper-\(legacyId)" })
    }

    /// Models grouped by family for UI display.
    static var modelsByFamily: [(family: ModelFamily, models: [ModelInfo])] {
        let grouped = Dictionary(grouping: availableModels, by: \.family)
        let order: [ModelFamily] = [.whisper, .moonshine, .senseVoice, .zipformer, .omnilingual, .parakeet]
        return order.compactMap { family in
            guard let models = grouped[family], !models.isEmpty else { return nil }
            return (family: family, models: models)
        }
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
