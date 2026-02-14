import Foundation

struct BackendResolution: Sendable {
    let card: ModelCard
    let requestedBackend: InferenceBackend
    let effectiveBackend: InferenceBackend
    let runtimeVariant: ModelRuntimeVariant?
    let fallbackReason: String?
}

enum BackendCapabilities {
    static func isBackendSupported(
        _ backend: InferenceBackend,
        platform: RuntimePlatform = .current
    ) -> Bool {
        switch backend {
        case .automatic, .legacy:
            return true
        case .cactus:
            switch platform {
            case .ios, .catalyst:
                return true
            case .macOS:
                return false
            }
        case .mlx:
            switch platform {
            case .macOS, .catalyst:
                return true
            case .ios:
                return false
            }
        }
    }
}

@MainActor
final class BackendResolver {
    static let shared = BackendResolver()

    private let logger = InferenceLogger.shared

    func resolve(
        card: ModelCard,
        requestedBackend: InferenceBackend,
        platform: RuntimePlatform = .current
    ) -> BackendResolution {
        let requested = requestedBackend == .automatic
            ? card.preferredBackend(for: platform)
            : requestedBackend

        let fallbackOrder = orderedFallbacks(
            requestedBackend: requested,
            platform: platform
        )

        for backend in fallbackOrder {
            guard BackendFeatureFlags.isBackendEnabled(backend) else {
                logger.log("[BackendResolver] backend disabled by feature flag: \(backend.rawValue)")
                continue
            }

            guard BackendCapabilities.isBackendSupported(backend, platform: platform) else {
                logger.log("[BackendResolver] backend unsupported on platform \(platform.rawValue): \(backend.rawValue)")
                continue
            }

            guard let variant = card.runtimeVariants.first(where: {
                $0.backend == backend && $0.isSupported(on: platform)
            }) else {
                logger.log("[BackendResolver] runtime variant unavailable for backend \(backend.rawValue), card=\(card.id)")
                continue
            }

            let reason: String?
            if backend == requested {
                reason = nil
            } else {
                reason = "Requested backend \(requested.rawValue) unavailable; using \(backend.rawValue)."
            }

            return BackendResolution(
                card: card,
                requestedBackend: requested,
                effectiveBackend: backend,
                runtimeVariant: variant,
                fallbackReason: reason
            )
        }

        let appleFallback = ModelInfo.availableModels.first(where: { $0.engineType == .appleSpeech })
        let fallbackVariant = appleFallback.map {
            ModelRuntimeVariant(
                id: "\($0.id)-legacy-fallback",
                backend: .legacy,
                engineType: .appleSpeech,
                runtimeLabel: $0.inferenceMethodLabel,
                platforms: RuntimePlatform.allCases,
                architectures: [],
                minimumOSVersion: nil,
                legacyModelId: $0.id,
                artifacts: [],
                isEnabled: true
            )
        }

        return BackendResolution(
            card: card,
            requestedBackend: requested,
            effectiveBackend: .legacy,
            runtimeVariant: fallbackVariant,
            fallbackReason: "No compatible runtime found for \(card.displayName). Using Apple Speech fallback."
        )
    }

    private func orderedFallbacks(
        requestedBackend: InferenceBackend,
        platform: RuntimePlatform
    ) -> [InferenceBackend] {
        var order: [InferenceBackend] = []

        func appendUnique(_ backend: InferenceBackend) {
            if !order.contains(backend) {
                order.append(backend)
            }
        }

        appendUnique(requestedBackend)

        switch platform {
        case .ios:
            appendUnique(.cactus)
            appendUnique(.legacy)
        case .macOS, .catalyst:
            appendUnique(.mlx)
            appendUnique(.legacy)
        }

        return order
    }
}
