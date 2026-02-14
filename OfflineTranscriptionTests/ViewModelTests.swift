import XCTest
@testable import OfflineTranscription

/// Tests for ViewModels.
@MainActor
final class ViewModelTests: XCTestCase {

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

    // MARK: - Iteration 1
    func testTranscriptionViewModelInitialState() {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        XCTAssertFalse(vm.isRecording)
        XCTAssertEqual(vm.confirmedText, "")
        XCTAssertEqual(vm.hypothesisText, "")
        XCTAssertEqual(vm.fullText, "")
        XCTAssertFalse(vm.showError)
        XCTAssertEqual(vm.errorMessage, "")
        XCTAssertFalse(vm.hasEngineError)
    }

    // MARK: - Iteration 2
    func testTranscriptionViewModelErrorOnStart() async {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        await vm.startRecording()
        XCTAssertTrue(vm.showError)
        XCTAssertFalse(vm.errorMessage.isEmpty)
    }

    // MARK: - Iteration 3
    func testModelManagementViewModelState() {
        let vm = ModelManagementViewModel(whisperService: WhisperService())
        XCTAssertFalse(vm.isDownloading)
        XCTAssertFalse(vm.isLoading)
        XCTAssertFalse(vm.isReady)
        XCTAssertEqual(vm.downloadProgress, 0.0)
        XCTAssertNil(vm.errorMessage)
        XCTAssertEqual(vm.selectedModel.cardId ?? vm.selectedModel.id, "whisper-base")
    }

    // MARK: - Iteration 4
    func testModelSelectionChange() {
        let service = WhisperService()
        let vm = ModelManagementViewModel(whisperService: service)
        let whisperTiny = ModelInfo.availableModels.first { $0.id == "whisper-tiny" }!
        vm.selectedModel = whisperTiny
        XCTAssertEqual(vm.selectedModel.id, "whisper-tiny")
        XCTAssertEqual(service.selectedModel.id, "whisper-tiny")
    }

    // MARK: - Iteration 6: ViewModel delegates to service
    func testViewModelDelegatesToService() {
        let service = WhisperService()
        let vm = TranscriptionViewModel(whisperService: service)
        service.testSetState(confirmedText: "test confirmed", hypothesisText: "test hypothesis")
        XCTAssertEqual(vm.confirmedText, "test confirmed")
        XCTAssertEqual(vm.hypothesisText, "test hypothesis")
    }

    // MARK: - Iteration 7: Toggle recording without model errors
    func testToggleRecordingErrors() async {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        await vm.toggleRecording()
        XCTAssertTrue(vm.showError)
    }

    // MARK: - Iteration 8: isModelDownloaded delegates to service
    func testIsModelDownloaded() {
        let service = WhisperService()
        let vm = ModelManagementViewModel(whisperService: service)
        for model in ModelInfo.availableModels {
            XCTAssertEqual(vm.isModelDownloaded(model), service.isModelDownloaded(model))
        }
    }
}
