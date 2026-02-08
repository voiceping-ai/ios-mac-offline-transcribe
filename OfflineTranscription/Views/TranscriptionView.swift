import SwiftUI

struct TranscriptionView: View {
    @Environment(WhisperService.self) private var whisperService
    @State private var viewModel: TranscriptionViewModel?
    @State private var showSettings = false
    @State private var recordingStartDate: Date?
    @State private var didAutoTest = false

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
                            }

                            if confirmedText.isEmpty && hypothesisText.isEmpty
                                && !vm.isRecording && !vm.isInterrupted
                            {
                                if vm.showPermissionDenied {
                                    permissionDeniedView(vm: vm)
                                } else {
                                    Text("Tap the microphone button to start transcribing.")
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
                .onChange(of: viewModel?.fullText ?? "") { _, _ in
                    guard viewModel?.isRecording == true else { return }
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
                .onChange(of: viewModel?.isRecording ?? false) { _, isRecording in
                    if isRecording {
                        recordingStartDate = Date()
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
                VStack(spacing: 2) {
                    Text("\(vm.selectedModel.displayName) · \(vm.selectedModel.languages)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(vm.selectedModel.inferenceMethodLabel)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
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

            // Controls
            HStack(spacing: 32) {
                #if DEBUG
                if let vm = viewModel, !vm.isRecording {
                    Button {
                        vm.transcribeTestFile("/tmp/test_speech.wav")
                    } label: {
                        Image(systemName: "doc.text.fill")
                            .font(.title2)
                    }
                    .accessibilityIdentifier("test_file_button")
                }
                #endif

                RecordButton(
                    isRecording: viewModel?.isRecording ?? false
                ) {
                    Task {
                        await viewModel?.toggleRecording()
                    }
                }

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gear")
                        .font(.title2)
                }
                .accessibilityIdentifier("settings_button")
            }
            .padding()
        }
        .navigationTitle("Transcribe")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        UIPasteboard.general.string = viewModel?.fullText ?? ""
                    } label: {
                        Label("Copy Text", systemImage: "doc.on.doc")
                    }
                    .disabled(viewModel == nil || (viewModel?.fullText.isEmpty ?? true))
                    Button(role: .destructive) {
                        viewModel?.clearTranscription()
                    } label: {
                        Label("Clear", systemImage: "trash")
                    }
                    .disabled(viewModel == nil || (viewModel?.fullText.isEmpty ?? true))
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .accessibilityIdentifier("overflow_menu")
            }
        }
        .sheet(isPresented: $showSettings) {
            ModelSettingsSheet()
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
            // Wait until model is loaded
            while whisperService.modelState != .loaded {
                try? await Task.sleep(for: .milliseconds(200))
            }
            guard !didAutoTest else { return }
            didAutoTest = true
            try? await Task.sleep(for: .milliseconds(500))
            let wavPath = Bundle.main.path(forResource: "test_speech", ofType: "wav") ?? "/tmp/test_speech.wav"
            viewModel?.transcribeTestFile(wavPath)
        }
        #endif
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

    var body: some View {
        NavigationStack {
            Form {
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

                ForEach(ModelInfo.modelsByFamily, id: \.family) { group in
                    Section(group.family.displayName) {
                        ForEach(group.models) { model in
                            Button {
                                isSwitching = true
                                Task {
                                    await whisperService.switchModel(to: model)
                                    isSwitching = false
                                    if whisperService.modelState == .loaded {
                                        dismiss()
                                    }
                                }
                            } label: {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(model.displayName)
                                        Text(model.description)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text("Inference: \(model.inferenceMethodLabel)")
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                    Spacer()
                                    VStack(alignment: .trailing) {
                                        Text(model.sizeOnDisk)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(model.languages)
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                            }
                            .accessibilityIdentifier("model_row_\(model.id)")
                            .disabled(
                                model.id == whisperService.selectedModel.id
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
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("settings_done_button")
                        .disabled(isSwitching)
                }
            }
        }
    }
}
