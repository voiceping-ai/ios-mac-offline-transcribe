import SwiftUI
#if os(macOS)
import AppKit
#endif

struct TranscriptionView: View {
    @Environment(WhisperService.self) private var whisperService
    @State private var viewModel: TranscriptionViewModel?
    @State private var showSettings = false

    @State private var recordingStartDate: Date?
    @State private var didAutoTest = false
    @State private var triggerBroadcast = false
    @State private var lastAutoScrollAt: Date = .distantPast

    private let autoScrollInterval: TimeInterval = 0.25
    private var isAutoTestMode: Bool {
        ProcessInfo.processInfo.arguments.contains("--auto-test")
    }

    var body: some View {
        VStack(spacing: 0) {
            // Transcription text area
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        if let vm = viewModel {
                            let confirmedText = vm.confirmedText
                                .trimmingCharacters(in: .whitespacesAndNewlines)
                            let hypothesisText = vm.hypothesisText
                                .trimmingCharacters(in: .whitespacesAndNewlines)

                            if !confirmedText.isEmpty {
                                Text(confirmedText)
                                    .font(.body)
                                    .foregroundStyle(.primary)
                                    .accessibilityIdentifier("confirmed_text")
                            }

                            if !hypothesisText.isEmpty {
                                Text(hypothesisText)
                                    .font(.body)
                                    .foregroundStyle(.secondary)
                                    .italic()
                                    .accessibilityIdentifier("hypothesis_text")
                            }

                            if confirmedText.isEmpty && hypothesisText.isEmpty
                                && !vm.isRecording && !vm.isInterrupted
                            {
                                if vm.showPermissionDenied {
                                    permissionDeniedView(vm: vm)
                                } else {
                                    Text(placeholderText)
                                        .font(.body)
                                        .foregroundStyle(.tertiary)
                                        .accessibilityIdentifier("idle_placeholder")
                                }
                            }

                            if vm.isRecording && confirmedText.isEmpty
                                && hypothesisText.isEmpty
                            {
                                Text("Listening...")
                                    .font(.body)
                                    .foregroundStyle(.tertiary)
                                    .accessibilityIdentifier("listening_text")
                            }

                            if vm.isInterrupted {
                                interruptedBanner
                            }
                        }

                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding()
                }
                .onChange(of: viewModel?.confirmedText ?? "") { _, _ in
                    guard viewModel?.isRecording == true else { return }
                    let now = Date()
                    guard now.timeIntervalSince(lastAutoScrollAt) >= autoScrollInterval else { return }
                    lastAutoScrollAt = now
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
                .onChange(of: viewModel?.hypothesisText ?? "") { _, _ in
                    guard viewModel?.isRecording == true else { return }
                    let now = Date()
                    guard now.timeIntervalSince(lastAutoScrollAt) >= autoScrollInterval else { return }
                    lastAutoScrollAt = now
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
                .onChange(of: viewModel?.isRecording ?? false) { _, isRecording in
                    if isRecording {
                        recordingStartDate = Date()
                        lastAutoScrollAt = Date()
                        proxy.scrollTo("bottom", anchor: .bottom)
                    } else {
                        recordingStartDate = nil
                    }
                }
            }

            Divider()

            // Audio visualizer
            if let vm = viewModel, vm.isRecording {
                AudioVisualizerView(energyLevels: vm.bufferEnergy)
                    .frame(height: 60)
                    .padding(.horizontal)
            }

            // Stats bar
            if let vm = viewModel, vm.isRecording {
                HStack {
                    Label(
                        FormatUtils.formatDuration(vm.bufferSeconds),
                        systemImage: "clock"
                    )
                    Spacer()
                    if vm.tokensPerSecond > 0 {
                        Text(String(format: "%.1f tok/s", vm.tokensPerSecond))
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
                .padding(.vertical, 4)
                .accessibilityIdentifier("stats_bar")
            }

            // Model info (always visible)
            if let vm = viewModel {
                VStack(spacing: 4) {
                    Text("\(vm.selectedModel.displayName) · \(vm.selectedModel.languages)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(whisperService.effectiveRuntimeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(Color.accentColor.opacity(0.12))
                        )
                    if let warning = whisperService.backendFallbackWarning {
                        Text(warning)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                .padding(.vertical, 4)
                .accessibilityIdentifier("model_info_label")
            }

            // Resource stats (always visible)
            if let vm = viewModel {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    let elapsed: Int = if let start = recordingStartDate, vm.isRecording {
                        Int(context.date.timeIntervalSince(start))
                    } else {
                        0
                    }
                    HStack(spacing: 16) {
                        if vm.isRecording {
                            Text("\(elapsed)s")
                        }
                        Text(String(format: "CPU %.0f%%", vm.cpuPercent))
                        Text(String(format: "RAM %.0f MB", vm.memoryMB))
                        if vm.tokensPerSecond > 0 {
                            Text(String(format: "%.1f tok/s", vm.tokensPerSecond))
                        }
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.vertical, 4)
            }

            // File transcription progress
            if let vm = viewModel, vm.isTranscribingFile {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Transcribing file...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .accessibilityIdentifier("file_transcribing_indicator")
            }

            #if os(iOS)
            // Audio source selector
            if viewModel != nil, !(viewModel?.isRecording ?? false) {
                @Bindable var service = whisperService
                Picker("Audio Source", selection: $service.audioCaptureMode) {
                    Label("Voice", systemImage: "mic.fill").tag(AudioCaptureMode.microphone)
                    Label("System", systemImage: "rectangle.dashed.badge.record").tag(AudioCaptureMode.systemBroadcast)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .accessibilityIdentifier("audio_source_picker")
            }

            // Hidden broadcast picker — must be in view hierarchy for system sheet
            if whisperService.audioCaptureMode == .systemBroadcast {
                BroadcastPickerView(triggerBroadcast: $triggerBroadcast)
                    .frame(width: 1, height: 1)
                    .opacity(0.01)
                    .allowsHitTesting(false)
            }
            #endif

            // Controls
            HStack(spacing: 32) {
                if let vm = viewModel, !vm.isRecording && !vm.isTranscribingFile {
                    Button {
                        let wavPath = Bundle.main.path(forResource: "test_speech", ofType: "wav") ?? "/tmp/test_speech.wav"
                        vm.transcribeTestFile(wavPath)
                    } label: {
                        Image(systemName: "doc.text.fill")
                            .font(.title2)
                    }
                    .accessibilityIdentifier("test_file_button")
                }

                RecordButton(
                    isRecording: viewModel?.isRecording ?? false
                ) {
                    if whisperService.audioCaptureMode == .systemBroadcast
                        && !(viewModel?.isRecording ?? false)
                        && !whisperService.isBroadcastActive {
                        triggerBroadcast = true
                    } else {
                        Task {
                            await viewModel?.toggleRecording()
                        }
                    }
                }
                .disabled(viewModel?.isTranscribingFile ?? false)

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gear")
                        .font(.title2)
                }
                .accessibilityIdentifier("settings_button")
            }
            .padding()

            AppVersionLabel()
                .padding(.bottom, 4)
        }
        .navigationTitle("Transcribe")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .overlay(alignment: .topTrailing) {
            #if DEBUG
            if isAutoTestMode, !whisperService.e2eOverlayPayload.isEmpty {
                VStack(alignment: .trailing, spacing: 4) {
                    Text("E2E Result Ready")
                        .font(.caption2)
                        .fontWeight(.semibold)
                    Text(whisperService.e2eOverlayPayload)
                        .font(.caption2.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .accessibilityIdentifier("e2e_overlay_payload")
                }
                .padding(8)
                .background(.black.opacity(0.72))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(8)
                .accessibilityIdentifier("e2e_overlay")
                .accessibilityLabel("e2e_overlay")
                .accessibilityValue(whisperService.e2eOverlayPayload)
            }
            #endif
        }
        .sheet(isPresented: $showSettings) {
            ModelSettingsSheet(
                fullText: viewModel?.fullText ?? "",
                onCopyText: {
                    #if os(macOS)
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(viewModel?.fullText ?? "", forType: .string)
                    #else
                    UIPasteboard.general.string = viewModel?.fullText ?? ""
                    #endif
                },
                onClearTranscription: {
                    viewModel?.clearTranscription()
                }
            )
        }
        .alert(
            "Error",
            isPresented: .init(
                get: { viewModel?.showError ?? false },
                set: { newValue in
                    guard viewModel != nil else { return }
                    viewModel?.showError = newValue
                }
            )
        ) {
            if viewModel?.showPermissionDenied == true {
                Button("Open Settings") { viewModel?.openSettings() }
                Button("Cancel", role: .cancel) {}
            } else {
                Button("OK", role: .cancel) {}
            }
        } message: {
            Text(viewModel?.errorMessage ?? "")
        }
        .onChange(of: whisperService.lastError != nil) { _, hasError in
            guard hasError, viewModel != nil else { return }
            viewModel?.surfaceEngineError()
        }
        .onAppear {
            if viewModel == nil {
                viewModel = TranscriptionViewModel(whisperService: whisperService)
            }
        }
        #if DEBUG
        .task {
            // Auto-test: wait for model to load, then transcribe test file
            guard ProcessInfo.processInfo.arguments.contains("--auto-test") else { return }
            let args = ProcessInfo.processInfo.arguments
            let argsDump = args.joined(separator: "\n")
            try? argsDump.write(
                to: URL(fileURLWithPath: "/tmp/ios_args_runtime.txt"),
                atomically: true,
                encoding: .utf8
            )
            let modelId = Self.autoTestModelId(from: args) ?? whisperService.selectedModel.id
            let timeoutSeconds = Self.autoTestLoadTimeoutSeconds(for: modelId)
            let deadline = Date().addingTimeInterval(timeoutSeconds)

            // Wait until model is loaded, but fail fast with an E2E result when setup stalls.
            while whisperService.modelState != .loaded && Date() < deadline {
                if whisperService.modelState == .error || whisperService.modelState == .unloaded {
                    break
                }
                try? await Task.sleep(for: .milliseconds(200))
            }
            guard whisperService.modelState == .loaded else {
                whisperService.writeE2EFailure(
                    reason: "model load failed/timed out for \(modelId) (state=\(whisperService.modelState.rawValue))"
                )
                return
            }
            guard !didAutoTest else { return }
            didAutoTest = true
            try? await Task.sleep(for: .milliseconds(500))
            let wavPath = Bundle.main.path(forResource: "test_speech", ofType: "wav") ?? "/tmp/test_speech.wav"
            viewModel?.transcribeTestFile(wavPath)
        }
        #endif
    }

    private static func autoTestModelId(from args: [String]) -> String? {
        guard let index = args.firstIndex(of: "--model-id"), index + 1 < args.count else {
            return nil
        }
        return args[index + 1]
    }

    private static func autoTestLoadTimeoutSeconds(for modelId: String) -> TimeInterval {
        if modelId.contains("large") || modelId.contains("omnilingual") { return 480 }
        if modelId == "whisper-small" || modelId == "parakeet-tdt-v3" { return 300 }
        if modelId == "whisper-base" { return 240 }
        return 120
    }

    // MARK: - Subviews

    private func permissionDeniedView(vm: TranscriptionViewModel) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "mic.slash.fill")
                .font(.system(size: 40))
                .foregroundStyle(.red)
            Text("Microphone access is required to transcribe speech.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Open Settings") {
                vm.openSettings()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    private var placeholderText: String {
        switch whisperService.audioCaptureMode {
        case .systemBroadcast:
            return "Start a system broadcast above, then audio from other apps will be transcribed."
        case .microphone:
            return "Tap the microphone button to start transcribing."
        }
    }

    private var interruptedBanner: some View {
        HStack {
            Image(systemName: "phone.fill")
            Text("Recording interrupted. Tap stop to finish.")
        }
        .font(.callout)
        .foregroundStyle(.orange)
        .padding(8)
        .frame(maxWidth: .infinity)
        .background(.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Settings Sheet

struct ModelSettingsSheet: View {
    @Environment(WhisperService.self) private var whisperService
    @Environment(\.dismiss) private var dismiss
    @State private var isSwitching = false

    let fullText: String
    let onCopyText: () -> Void
    let onClearTranscription: () -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Actions") {
                    Button {
                        onCopyText()
                    } label: {
                        Label("Copy Text", systemImage: "doc.on.doc")
                    }
                    .accessibilityIdentifier("settings_copy_text")
                    .disabled(fullText.isEmpty)

                    Button(role: .destructive) {
                        onClearTranscription()
                    } label: {
                        Label("Clear Transcription", systemImage: "trash")
                    }
                    .accessibilityIdentifier("settings_clear_transcription")
                    .disabled(fullText.isEmpty)
                }

                Section("Current Model") {
                    HStack {
                        Text(whisperService.selectedModel.displayName)
                            .accessibilityIdentifier("settings_current_model")
                        Spacer()
                        Text(whisperService.selectedModel.parameterCount)
                            .foregroundStyle(.secondary)
                    }
                    Text("Inference: \(whisperService.selectedModel.inferenceMethodLabel)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Requested backend: \(whisperService.selectedInferenceBackend.displayName)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Text("Effective backend: \(whisperService.effectiveInferenceBackend.displayName)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    if let warning = whisperService.backendFallbackWarning {
                        Text(warning)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }

                if BackendFeatureFlags.isBackendSelectorEnabled,
                   let selectedCard = whisperService.selectedCard(),
                   whisperService.availableBackends(for: selectedCard).count > 1 {
                    Section("Backend") {
                        Picker(
                            "Backend",
                            selection: Binding(
                                get: { whisperService.selectedInferenceBackend },
                                set: { whisperService.setSelectedInferenceBackend($0) }
                            )
                        ) {
                            ForEach(whisperService.availableBackends(for: selectedCard), id: \.self) { backend in
                                Text(backend.displayName).tag(backend)
                            }
                        }
                    }
                }

                if isSwitching {
                    Section {
                        if whisperService.modelState == .downloading {
                            VStack(spacing: 8) {
                                ProgressView(
                                    value: whisperService.downloadProgress
                                ) {
                                    Text(
                                        "Downloading \(whisperService.selectedModel.displayName)..."
                                    )
                                    .font(.subheadline)
                                }
                                Text(
                                    "\(Int(whisperService.downloadProgress * 100))%"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                        } else {
                            HStack {
                                ProgressView()
                                Text("Loading model...")
                                    .padding(.leading, 8)
                            }
                        }
                    }
                }

                if !isSwitching, let error = whisperService.lastError {
                    Section {
                        Text(error.localizedDescription)
                            .foregroundStyle(.red)
                            .font(.callout)
                    }
                }

                ForEach(whisperService.modelCardsByFamily, id: \.family) { group in
                    Section(group.family.displayName) {
                        ForEach(group.cards) { card in
                            Button {
                                isSwitching = true
                                whisperService.setSelectedModelCard(card.id)
                                let supportedBackends = whisperService.availableBackends(for: card)
                                if !supportedBackends.contains(whisperService.selectedInferenceBackend) {
                                    whisperService.setSelectedInferenceBackend(card.preferredBackend())
                                }
                                Task {
                                    await whisperService.setupModel()
                                    isSwitching = false
                                    if whisperService.modelState == .loaded {
                                        dismiss()
                                    }
                                }
                            } label: {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(card.displayName)
                                        Text(card.description)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        let backend = card.id == whisperService.selectedModelCardId
                                            ? whisperService.selectedInferenceBackend
                                            : card.preferredBackend()
                                        Text("Inference: \(whisperService.runtimeLabel(for: card, requestedBackend: backend))")
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                    Spacer()
                                    VStack(alignment: .trailing) {
                                        Text(card.sizeOnDisk)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(card.languages)
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                            }
                            .accessibilityIdentifier("model_row_\(card.id)")
                            .disabled(
                                card.id == whisperService.selectedModelCardId
                                || isSwitching
                            )
                        }
                    }
                }

                Section("Transcription Settings") {
                    @Bindable var service = whisperService
                    Toggle("Voice Activity Detection", isOn: $service.useVAD)
                        .accessibilityIdentifier("vad_toggle")
                    Toggle("Enable Timestamps", isOn: $service.enableTimestamps)
                        .accessibilityIdentifier("timestamps_toggle")
                }

                Section {
                    AppVersionLabel()
                        .frame(maxWidth: .infinity)
                }
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("settings_done_button")
                        .disabled(isSwitching)
                }
                #else
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("settings_done_button")
                        .disabled(isSwitching)
                }
                #endif
            }
        }
    }
}
