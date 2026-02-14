import SwiftUI

struct ModelSetupView: View {
    @Environment(WhisperService.self) private var whisperService
    @State private var viewModel: ModelManagementViewModel?

    private var isBusy: Bool {
        whisperService.modelState == .downloading
            || whisperService.modelState == .downloaded
            || whisperService.modelState == .loading
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                headerSection
                modelPickerSection
                Spacer()
                statusSection
            }
            .accessibilityIdentifier("model_setup_view")
            .navigationTitle("Setup")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
        .task {
            if viewModel == nil {
                viewModel = ModelManagementViewModel(whisperService: whisperService)
            }
            await whisperService.refreshModelCatalog()
        }
    }

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 72))
                .foregroundStyle(.blue)
            Text("Offline Transcription")
                .font(.largeTitle.bold())
                .accessibilityIdentifier("setup_title")
            Text(
                "Download a speech recognition model to get started. Models are stored on-device for fully offline use."
            )
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .padding(.horizontal)
        }
        .padding(.top, 40)
    }

    private var modelPickerSection: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                ForEach(whisperService.modelCardsByFamily, id: \.family) { group in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(group.family.displayName)
                            .font(.headline)
                            .foregroundStyle(.secondary)

                        ForEach(group.cards) { card in
                            modelRow(card)
                        }
                    }
                }
            }
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private func modelRow(_ card: ModelCard) -> some View {
        let isSelectedCard = whisperService.selectedModelCardId == card.id
        let availableBackends = whisperService.availableBackends(for: card)
        let selectedBackend = isSelectedCard
            ? whisperService.selectedInferenceBackend
            : card.preferredBackend()
        let resolvedModel = whisperService.resolvedModelInfo(
            for: card,
            requestedBackend: selectedBackend
        )
        let downloaded = resolvedModel.map { viewModel?.isModelDownloaded($0) ?? false } ?? false

        ModelPickerRow(
            card: card,
            isSelected: isSelectedCard,
            selectedBackend: selectedBackend,
            availableBackends: availableBackends,
            effectiveBackend: isSelectedCard
                ? whisperService.effectiveInferenceBackend
                : selectedBackend,
            effectiveRuntimeLabel: isSelectedCard
                ? whisperService.effectiveRuntimeLabel
                : whisperService.runtimeLabel(for: card, requestedBackend: selectedBackend),
            fallbackWarning: isSelectedCard ? whisperService.backendFallbackWarning : nil,
            isDownloaded: downloaded,
            isDownloading: whisperService.modelState == .downloading && isSelectedCard,
            downloadProgress: whisperService.downloadProgress,
            isLoading: whisperService.modelState == .loading && isSelectedCard,
            onBackendChange: { backend in
                whisperService.setSelectedModelCard(card.id)
                whisperService.setSelectedInferenceBackend(backend)
            },
            onTap: {
                whisperService.setSelectedModelCard(card.id)
                Task {
                    await viewModel?.downloadAndSetup()
                }
            }
        )
        .disabled(isBusy)
    }

    @ViewBuilder
    private var statusSection: some View {
        VStack(spacing: 16) {
            if whisperService.modelState != .downloading && whisperService.modelState != .loading {
                Text("Tap a model to download and get started.")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .accessibilityIdentifier("setup_prompt")
            }

            if let warning = whisperService.backendFallbackWarning {
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.horizontal)
            }

            if let error = whisperService.lastError {
                Text(error.localizedDescription)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }
        }
        .padding(.bottom, 8)

        AppVersionLabel()
            .padding(.bottom, 16)
    }
}
