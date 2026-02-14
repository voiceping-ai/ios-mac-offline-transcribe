import SwiftUI
#if os(iOS)
import UIKit
#endif

@main
struct OfflineTranscriptionApp: App {
    let whisperService: WhisperService

    init() {
        // Clear persisted state BEFORE WhisperService.init() reads UserDefaults
        if ProcessInfo.processInfo.arguments.contains("--reset-state") {
            UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
            UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
            UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
        }
        whisperService = WhisperService()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(whisperService)
                #if os(macOS)
                .frame(minWidth: 500, minHeight: 600)
                #endif
        }
        #if os(macOS)
        Settings {
            Text("Preferences will be available in a future release.")
                .padding()
        }
        #endif
    }
}

struct RootView: View {
    @Environment(WhisperService.self) private var whisperService

    private static var autoTestModelId: String? {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--model-id"), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    private static var isAutoTestRun: Bool {
        ProcessInfo.processInfo.arguments.contains("--auto-test")
    }

    var body: some View {
        Group {
            switch whisperService.modelState {
            case .loaded:
                TranscriptionRootView()
            case .loading, .downloading, .downloaded:
                VStack(spacing: 8) {
                    ProgressView(whisperService.modelState == .downloading
                        ? "Downloading model..." : "Loading model...")
                    if whisperService.modelState == .downloading {
                        Text("\(Int(whisperService.downloadProgress * 100))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if !whisperService.loadingStatusMessage.isEmpty {
                        Text(whisperService.loadingStatusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    AppVersionLabel()
                }
            default:
                // Only keep TranscriptionRootView if a model was previously loaded
                // (e.g. during model switch). Prevents navigating to transcription
                // when download/load failed and no model is ready.
                if whisperService.activeEngine?.modelState == .loaded {
                    TranscriptionRootView()
                } else {
                    ModelSetupView()
                }
            }
        }
        .task {
            let resetState = ProcessInfo.processInfo.arguments.contains("--reset-state")
            configureIdleTimerForUITests()

            // --reset-state: clear UserDefaults for clean UI test runs
            if resetState {
                UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
                UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
                UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
            }

            // Auto-load only for E2E / UI testing (--model-id / --auto-test).
            // Normal launch always starts on the model selection screen.
            if let modelId = Self.autoTestModelId,
               let model = ModelInfo.availableModels.first(where: { $0.id == modelId }) {
                await whisperService.switchModel(to: model)
            } else if Self.isAutoTestRun {
                await whisperService.loadModelIfAvailable()
            }
        }
    }

    private func configureIdleTimerForUITests() {
        #if os(iOS)
        guard Self.isAutoTestRun else { return }
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
    }
}

struct TranscriptionRootView: View {
    var body: some View {
        NavigationStack {
            TranscriptionView()
        }
        .accessibilityIdentifier("main_tab_view")
    }
}
