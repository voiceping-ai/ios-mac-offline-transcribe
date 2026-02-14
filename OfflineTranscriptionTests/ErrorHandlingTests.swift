import XCTest
@testable import OfflineTranscription

/// Tests for error handling, permission flows, and error propagation.
@MainActor
final class ErrorHandlingTests: XCTestCase {

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

    // MARK: - AppError Coverage

    func testAllAppErrorCases() {
        let err = NSError(domain: "test", code: 42)

        let cases: [AppError] = [
            .microphonePermissionDenied,
            .microphonePermissionRestricted,
            .modelDownloadFailed(underlying: err),
            .modelLoadFailed(underlying: err),
            .transcriptionFailed(underlying: err),
            .audioSessionSetupFailed(underlying: err),
            .noModelSelected,
            .modelNotReady,
            .audioInterrupted,
        ]

        for appError in cases {
            XCTAssertFalse(
                appError.localizedDescription.isEmpty,
                "Error \(appError) has empty description"
            )
        }
    }

    func testAudioInterruptedErrorDescription() {
        let err = AppError.audioInterrupted
        XCTAssertTrue(err.localizedDescription.contains("interrupted"))
    }

    func testMicrophonePermissionDeniedContainsSettings() {
        let err = AppError.microphonePermissionDenied
        XCTAssertTrue(err.localizedDescription.contains("Settings"))
    }

    func testModelNotReadyDescription() {
        let err = AppError.modelNotReady
        XCTAssertTrue(err.localizedDescription.contains("not ready"))
    }

    func testUnderlyingErrorPreserved() {
        let underlying = NSError(domain: "com.test", code: 99, userInfo: [
            NSLocalizedDescriptionKey: "Custom underlying error"
        ])
        let err = AppError.transcriptionFailed(underlying: underlying)
        XCTAssertTrue(err.localizedDescription.contains("Custom underlying error"))
    }

    // MARK: - WhisperService Error State

    func testLastErrorInitiallyNil() {
        let s = WhisperService()
        XCTAssertNil(s.lastError)
    }

    func testClearLastError() {
        let s = WhisperService()
        s.clearLastError()
        XCTAssertNil(s.lastError)
    }

    func testClearTranscriptionClearsError() {
        let s = WhisperService()
        s.clearTranscription()
        XCTAssertNil(s.lastError)
    }

    // MARK: - ViewModel Error Bridging

    func testViewModelShowErrorInitiallyFalse() {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        XCTAssertFalse(vm.showError)
        XCTAssertEqual(vm.errorMessage, "")
    }

    func testViewModelHasEngineError() {
        let s = WhisperService()
        let vm = TranscriptionViewModel(whisperService: s)
        XCTAssertFalse(vm.hasEngineError)
    }

    func testViewModelStartRecordingShowsError() async {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        await vm.startRecording()
        XCTAssertTrue(vm.showError)
        XCTAssertFalse(vm.errorMessage.isEmpty)
    }

    func testViewModelPermissionDeniedDetection() async {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        XCTAssertFalse(vm.showPermissionDenied)
    }

    func testViewModelClearResetsPermissionDenied() {
        let vm = TranscriptionViewModel(whisperService: WhisperService())
        vm.showPermissionDenied = true
        vm.clearTranscription()
        XCTAssertFalse(vm.showPermissionDenied)
    }

    // MARK: - Model State Errors

    func testStartRecordingWithoutModelThrowsModelNotReady() async {
        let s = WhisperService()
        do {
            try await s.startRecording()
            XCTFail("Expected modelNotReady error")
        } catch let error as AppError {
            if case .modelNotReady = error {
                // Expected
            } else {
                XCTFail("Expected modelNotReady, got \(error)")
            }
        } catch {
            XCTFail("Expected AppError, got \(error)")
        }
    }

    func testSessionStateResetAfterStartFailure() async {
        let s = WhisperService()
        do {
            try await s.startRecording()
        } catch {
            // Expected
        }
        XCTAssertEqual(s.sessionState, .idle, "Should reset to idle after failure")
    }
}
