import Foundation

/// Resolves backend-aware model cards from remote, cache, bundle, then legacy static catalog.
@MainActor
final class ModelCatalogService {
    static let shared = ModelCatalogService()

    static let manifestURLDefaultsKey = "modelCatalogManifestURL"
    private static let schemaVersion = 1

    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private let session: URLSession
    private let logger = InferenceLogger.shared

    init(session: URLSession = .shared) {
        self.session = session
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    func loadCatalog() async -> ModelCatalog {
        if let remoteURL = configuredManifestURL() {
            do {
                let remoteCatalog = try await loadRemoteCatalog(from: remoteURL)
                try cacheCatalog(remoteCatalog)
                return remoteCatalog
            } catch {
                logger.log("[ModelCatalogService] Remote manifest load failed: \(error.localizedDescription)")
            }
        }

        if let cachedCatalog = loadCachedCatalog() {
            return cachedCatalog
        }

        if let bundledCatalog = loadBundledCatalog() {
            return bundledCatalog
        }

        return ModelCatalog(source: .legacy, cards: ModelInfo.legacyModelCards)
    }

    func loadLocalFallbackCatalog() -> ModelCatalog {
        if let cachedCatalog = loadCachedCatalog() {
            return cachedCatalog
        }
        if let bundledCatalog = loadBundledCatalog() {
            return bundledCatalog
        }
        return ModelCatalog(source: .legacy, cards: ModelInfo.legacyModelCards)
    }

    func configuredManifestURL() -> URL? {
        let defaults = UserDefaults.standard
        if let rawValue = defaults.string(forKey: Self.manifestURLDefaultsKey),
           let url = URL(string: rawValue),
           !rawValue.isEmpty {
            return url
        }
        if let envURL = ProcessInfo.processInfo.environment["MODEL_CATALOG_URL"],
           let url = URL(string: envURL),
           !envURL.isEmpty {
            return url
        }
        return nil
    }

    private func loadRemoteCatalog(from url: URL) async throws -> ModelCatalog {
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NSError(
                domain: "ModelCatalogService",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Manifest request failed with non-2xx response"]
            )
        }

        let manifest = try decodeAndValidateManifest(data)
        return ModelCatalog(source: .remote, cards: gateVariants(in: manifest.cards))
    }

    private func loadCachedCatalog() -> ModelCatalog? {
        guard let cacheURL = cacheFileURL(),
              FileManager.default.fileExists(atPath: cacheURL.path),
              let data = try? Data(contentsOf: cacheURL),
              let manifest = try? decodeAndValidateManifest(data) else {
            return nil
        }

        return ModelCatalog(source: .cached, cards: gateVariants(in: manifest.cards))
    }

    private func loadBundledCatalog() -> ModelCatalog? {
        guard let url = bundledManifestURL(),
              let data = try? Data(contentsOf: url),
              let manifest = try? decodeAndValidateManifest(data) else {
            return nil
        }

        return ModelCatalog(source: .bundled, cards: gateVariants(in: manifest.cards))
    }

    private func cacheCatalog(_ catalog: ModelCatalog) throws {
        let manifest = ModelCatalogManifest(schemaVersion: Self.schemaVersion, cards: catalog.cards)
        let data = try encoder.encode(manifest)

        guard let cacheURL = cacheFileURL() else { return }
        let directory = cacheURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let tempURL = directory.appendingPathComponent("manifest-\(UUID().uuidString).json")
        try data.write(to: tempURL, options: .atomic)

        if FileManager.default.fileExists(atPath: cacheURL.path) {
            try FileManager.default.removeItem(at: cacheURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: cacheURL)
    }

    private func decodeAndValidateManifest(_ data: Data) throws -> ModelCatalogManifest {
        let manifest = try decoder.decode(ModelCatalogManifest.self, from: data)

        guard manifest.schemaVersion == Self.schemaVersion else {
            throw NSError(
                domain: "ModelCatalogService",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported manifest schema \(manifest.schemaVersion)"]
            )
        }

        for card in manifest.cards {
            guard !card.id.isEmpty else {
                throw NSError(
                    domain: "ModelCatalogService",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: "Manifest card id must not be empty"]
                )
            }
            for variant in card.runtimeVariants {
                for artifact in variant.artifacts {
                    guard !artifact.checksum.isEmpty,
                          artifact.sizeBytes >= 0,
                          !artifact.relativePath.contains("..") else {
                        throw NSError(
                            domain: "ModelCatalogService",
                            code: -4,
                            userInfo: [NSLocalizedDescriptionKey: "Invalid artifact metadata for card \(card.id)"]
                        )
                    }
                }
            }
        }

        return manifest
    }

    private func gateVariants(in cards: [ModelCard]) -> [ModelCard] {
        cards.compactMap { card in
            let variants = card.runtimeVariants.filter { variant in
                guard variant.isSupported() else { return false }
                guard BackendFeatureFlags.isBackendEnabled(variant.backend) else { return false }
                guard BackendCapabilities.isBackendSupported(variant.backend) else { return false }
                if let legacyModelId = variant.legacyModelId {
                    let isSupportedLegacy = ModelInfo.supportedModels.contains(where: {
                        $0.id == legacyModelId
                    })
                    if !isSupportedLegacy {
                        return false
                    }
                }
                return true
            }

            guard !variants.isEmpty else { return nil }

            return ModelCard(
                id: card.id,
                displayName: card.displayName,
                parameterCount: card.parameterCount,
                sizeOnDisk: card.sizeOnDisk,
                description: card.description,
                family: card.family,
                languages: card.languages,
                runtimeVariants: variants
            )
        }
    }

    private func cacheFileURL() -> URL? {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            return nil
        }

        return appSupport
            .appendingPathComponent("ModelCatalog", isDirectory: true)
            .appendingPathComponent("manifest-cache.json")
    }

    private func bundledManifestURL() -> URL? {
        Bundle.main.url(
            forResource: "model-catalog.default",
            withExtension: "json",
            subdirectory: "Resources"
        ) ?? Bundle.main.url(
            forResource: "model-catalog.default",
            withExtension: "json"
        )
    }
}
