import XCTest
@testable import OfflineTranscription

/// Tests for WhisperService state, models, utilities, and error handling.
@MainActor
final class WhisperServiceTests: XCTestCase {

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
        UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
        UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
        UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
        UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
        super.tearDown()
    }

    private func makeSegment(text: String, start: Float, end: Float) -> ASRSegment {
        ASRSegment(id: Int.random(in: 0...10000), text: text, start: start, end: end)
    }

    // MARK: - Iteration 1
    func testInitialState() {
        let s = WhisperService()
        XCTAssertNil(s.whisperKit)
        XCTAssertEqual(s.modelState, .unloaded)
        XCTAssertFalse(s.isRecording)
        XCTAssertFalse(s.isTranscribing)
        XCTAssertEqual(s.confirmedText, "")
        XCTAssertEqual(s.hypothesisText, "")
        XCTAssertEqual(s.confirmedSegments.count, 0)
        XCTAssertEqual(s.unconfirmedSegments.count, 0)
        XCTAssertEqual(s.bufferEnergy.count, 0)
        XCTAssertEqual(s.bufferSeconds, 0.0)
        XCTAssertEqual(s.tokensPerSecond, 0.0)
        XCTAssertEqual(s.downloadProgress, 0.0)
        XCTAssertNil(s.lastError)
    }

    // MARK: - Iteration 2
    func testDefaultModelSelection() {
        let s = WhisperService()
        XCTAssertEqual(s.selectedModelCardId, "whisper-base")
        XCTAssertEqual(s.selectedModel.cardId ?? s.selectedModel.id, "whisper-base")
        XCTAssertEqual(s.selectedModel.displayName, "Whisper Base")
        XCTAssertEqual(s.selectedModel.family, .whisper)
    }

    // MARK: - Iteration 3
    func testModelInfoCatalog() {
        let models = ModelInfo.availableModels
        XCTAssertEqual(models.count, 15)
        XCTAssertEqual(models[0].id, "whisper-tiny")
        XCTAssertEqual(models[1].id, "whisper-base")
        XCTAssertEqual(models[2].id, "whisper-small")
        XCTAssertEqual(models[3].id, "whisper-large-v3-turbo")
        XCTAssertEqual(models[4].id, "whisper-large-v3-turbo-compressed")
        XCTAssertEqual(models[5].id, "moonshine-tiny")
        XCTAssertEqual(models[6].id, "moonshine-base")
        XCTAssertEqual(models[7].id, "sensevoice-small")
        XCTAssertEqual(models[8].id, "zipformer-20m")
        XCTAssertEqual(models[9].id, "omnilingual-300m")
        XCTAssertEqual(models[10].id, "parakeet-tdt-v3")
        XCTAssertEqual(models[11].id, "apple-speech")
        XCTAssertEqual(models[12].id, "qwen3-asr-0.6b")
        XCTAssertEqual(models[13].id, "qwen3-asr-0.6b-mlx")
        XCTAssertEqual(models[14].id, "qwen3-asr-0.6b-onnx")
        XCTAssertEqual(ModelInfo.defaultModel.id, "whisper-base")
    }

    // MARK: - Iteration 4
    func testStartRecordingWithoutModelThrows() async {
        let s = WhisperService()
        do {
            try await s.startRecording()
            XCTFail("Expected modelNotReady error")
        } catch let error as AppError {
            XCTAssertEqual(error.localizedDescription, AppError.modelNotReady.localizedDescription)
        } catch {
            XCTFail("Expected AppError, got \(error)")
        }
    }

    // MARK: - Iteration 5
    func testClearTranscriptionResetsState() {
        let s = WhisperService()
        s.testSetState(
            confirmedText: "Hello",
            hypothesisText: "world",
            confirmedSegments: [makeSegment(text: "Hello", start: 0, end: 1)],
            unconfirmedSegments: [makeSegment(text: "world", start: 1, end: 2)]
        )
        s.clearTranscription()
        XCTAssertEqual(s.confirmedText, "")
        XCTAssertEqual(s.hypothesisText, "")
        XCTAssertEqual(s.confirmedSegments.count, 0)
        XCTAssertEqual(s.unconfirmedSegments.count, 0)
        XCTAssertFalse(s.isRecording)
        XCTAssertEqual(s.bufferSeconds, 0.0)
        XCTAssertEqual(s.tokensPerSecond, 0.0)
    }

    // MARK: - Iteration 6
    func testFullTranscriptionText() {
        let s = WhisperService()
        XCTAssertEqual(s.fullTranscriptionText, "")

        s.testSetState(confirmedSegments: [makeSegment(text: "Hello", start: 0, end: 1)])
        XCTAssertEqual(s.fullTranscriptionText, "Hello")

        s.testSetState(
            confirmedSegments: [makeSegment(text: "Hello", start: 0, end: 1)],
            unconfirmedSegments: [makeSegment(text: " world", start: 1, end: 2)]
        )
        XCTAssertEqual(s.fullTranscriptionText, "Hello world")
    }

    // MARK: - Iteration 7
    func testFormatDuration() {
        XCTAssertEqual(FormatUtils.formatDuration(0), "0:00")
        XCTAssertEqual(FormatUtils.formatDuration(5), "0:05")
        XCTAssertEqual(FormatUtils.formatDuration(60), "1:00")
        XCTAssertEqual(FormatUtils.formatDuration(65), "1:05")
        XCTAssertEqual(FormatUtils.formatDuration(3600), "1:00:00")
        XCTAssertEqual(FormatUtils.formatDuration(3661), "1:01:01")
        XCTAssertEqual(FormatUtils.formatDuration(0.5), "0:00")
    }

    // MARK: - Iteration 8
    func testFormatFileSize() {
        let mb = FormatUtils.formatFileSize(80_000_000)
        XCTAssertTrue(mb.contains("MB"), "Got: \(mb)")
        let gb = FormatUtils.formatFileSize(1_000_000_000)
        XCTAssertTrue(gb.contains("GB") || gb.contains("MB"), "Got: \(gb)")
    }

    // MARK: - Iteration 9
    func testAppErrorDescriptions() {
        XCTAssertTrue(AppError.microphonePermissionDenied.localizedDescription.contains("denied"))
        XCTAssertTrue(AppError.microphonePermissionRestricted.localizedDescription.contains("restricted"))
        XCTAssertTrue(AppError.modelNotReady.localizedDescription.contains("not ready"))
        XCTAssertTrue(AppError.noModelSelected.localizedDescription.contains("No transcription"))

        let err = NSError(domain: "t", code: 1)
        XCTAssertTrue(AppError.modelDownloadFailed(underlying: err).localizedDescription.contains("download"))
        XCTAssertTrue(AppError.modelLoadFailed(underlying: err).localizedDescription.contains("load"))
        XCTAssertTrue(AppError.transcriptionFailed(underlying: err).localizedDescription.contains("Transcription"))
    }

    // MARK: - Iteration 10 (TranscriptionRecord removed from this target)

    // MARK: - New: Model families & engine types
    func testModelFamilies() {
        let whisperModels = ModelInfo.availableModels.filter { $0.family == .whisper }
        XCTAssertEqual(whisperModels.count, 5)
        XCTAssertTrue(whisperModels.allSatisfy { $0.engineType == .whisperKit })

        let moonshineModels = ModelInfo.availableModels.filter { $0.family == .moonshine }
        XCTAssertEqual(moonshineModels.count, 2)
        XCTAssertTrue(moonshineModels.allSatisfy { $0.engineType == .sherpaOnnxOffline })

        let senseVoiceModels = ModelInfo.availableModels.filter { $0.family == .senseVoice }
        XCTAssertEqual(senseVoiceModels.count, 1)

        let zipformerModels = ModelInfo.availableModels.filter { $0.family == .zipformer }
        XCTAssertEqual(zipformerModels.count, 1)
        XCTAssertTrue(zipformerModels.allSatisfy { $0.engineType == .sherpaOnnxStreaming })

        let omniModels = ModelInfo.availableModels.filter { $0.family == .omnilingual }
        XCTAssertEqual(omniModels.count, 1)
        XCTAssertTrue(omniModels.allSatisfy { $0.engineType == .sherpaOnnxOffline })

        let parakeetModels = ModelInfo.availableModels.filter { $0.family == .parakeet }
        XCTAssertEqual(parakeetModels.count, 1)
        XCTAssertTrue(parakeetModels.allSatisfy { $0.engineType == .fluidAudio })

        let qwenModels = ModelInfo.availableModels.filter { $0.family == .qwenASR }
        XCTAssertEqual(qwenModels.count, 3)
        XCTAssertTrue(qwenModels.contains { $0.engineType == .qwenASR })
        XCTAssertTrue(qwenModels.contains { $0.engineType == .mlx })
        XCTAssertTrue(qwenModels.contains { $0.engineType == .qwenOnnx })
    }

    func testLegacyModelIdLookup() {
        XCTAssertEqual(ModelInfo.findByLegacyId("tiny")?.id, "whisper-tiny")
        XCTAssertEqual(ModelInfo.findByLegacyId("base")?.id, "whisper-base")
        XCTAssertEqual(ModelInfo.findByLegacyId("small")?.id, "whisper-small")
        XCTAssertEqual(ModelInfo.findByLegacyId("whisper-base")?.id, "whisper-base")
        XCTAssertNil(ModelInfo.findByLegacyId("nonexistent"))
    }

    func testModelsByFamily() {
        let groups = ModelInfo.modelsByFamily
        XCTAssertEqual(groups.count, 8)
        let families = Set(groups.map(\.family))
        XCTAssertEqual(
            families,
            [.whisper, .moonshine, .senseVoice, .zipformer, .omnilingual, .parakeet, .appleSpeech, .qwenASR]
        )
    }
}
