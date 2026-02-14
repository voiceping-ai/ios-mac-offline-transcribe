import SwiftUI

struct ModelPickerRow: View {
    let card: ModelCard
    let isSelected: Bool
    let selectedBackend: InferenceBackend
    let availableBackends: [InferenceBackend]
    let effectiveBackend: InferenceBackend
    let effectiveRuntimeLabel: String
    let fallbackWarning: String?
    let isDownloaded: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let isLoading: Bool
    let onBackendChange: (InferenceBackend) -> Void
    let onTap: () -> Void

    private var defaultCellBackground: Color {
        #if os(macOS)
        Color(.controlBackgroundColor)
        #else
        Color(.secondarySystemBackground)
        #endif
    }

    private var clampedProgress: Double {
        min(max(downloadProgress, 0.0), 1.0)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: onTap) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(card.displayName)
                                .font(.headline)
                            Text(card.parameterCount)
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.fill.tertiary)
                                .clipShape(Capsule())
                            if isDownloaded {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                    .font(.caption)
                            }
                        }

                        Text(card.description)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Text("Runtime: \(effectiveRuntimeLabel)")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Text("Size: \(card.sizeOnDisk)")
                            .font(.caption)
                            .foregroundStyle(.tertiary)

                        if let fallbackWarning, isSelected {
                            Text(fallbackWarning)
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }

                        if isDownloading {
                            ProgressView(value: clampedProgress)
                                .tint(.blue)
                                .padding(.top, 4)
                            Text("Downloading \(Int(clampedProgress * 100))%")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else if isLoading {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .controlSize(.small)
                                Text("Loading model...")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.top, 4)
                        }
                    }

                    Spacer()

                    Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                        .foregroundStyle(isSelected ? .blue : .secondary)
                        .font(.title3)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(
                            isSelected
                                ? Color.blue.opacity(0.08)
                                : defaultCellBackground
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isSelected ? Color.blue.opacity(0.3) : Color.clear, lineWidth: 1.5)
                )
            }
            .buttonStyle(.plain)

            if BackendFeatureFlags.isBackendSelectorEnabled,
               availableBackends.count > 1 {
                backendPicker
                    .padding(.horizontal, 8)
            }
        }
    }

    private var backendPicker: some View {
        HStack(spacing: 8) {
            Text("Backend")
                .font(.caption2)
                .foregroundStyle(.secondary)

            Picker("Backend", selection: backendBinding) {
                ForEach(availableBackends, id: \.self) { backend in
                    Text(backend.displayName)
                        .tag(backend)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()

            if isSelected {
                Text("Effective: \(effectiveBackend.displayName)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var backendBinding: Binding<InferenceBackend> {
        Binding(
            get: { selectedBackend },
            set: { onBackendChange($0) }
        )
    }
}
