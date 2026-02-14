import Foundation

enum InferenceBackend: String, Codable, CaseIterable, Sendable, Hashable {
    case automatic
    case legacy
    case cactus
    case mlx

    var displayName: String {
        switch self {
        case .automatic:
            return "Automatic"
        case .legacy:
            return "Legacy"
        case .cactus:
            return "Cactus"
        case .mlx:
            return "MLX"
        }
    }
}

enum RuntimePlatform: String, Codable, CaseIterable, Sendable, Hashable {
    case ios
    case macOS
    case catalyst

    static var current: RuntimePlatform {
        #if targetEnvironment(macCatalyst)
        return .catalyst
        #elseif os(macOS)
        return .macOS
        #else
        return .ios
        #endif
    }

    static var currentArchitecture: String {
        #if arch(arm64)
        return "arm64"
        #elseif arch(x86_64)
        return "x86_64"
        #else
        return "unknown"
        #endif
    }
}

enum ArtifactChecksumAlgorithm: String, Codable, Sendable, Hashable {
    case sha256
}

struct ModelArtifact: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let relativePath: String
    let url: URL
    let checksum: String
    let checksumAlgorithm: ArtifactChecksumAlgorithm
    let sizeBytes: Int64

    init(
        id: String,
        relativePath: String,
        url: URL,
        checksum: String,
        checksumAlgorithm: ArtifactChecksumAlgorithm = .sha256,
        sizeBytes: Int64
    ) {
        self.id = id
        self.relativePath = relativePath
        self.url = url
        self.checksum = checksum
        self.checksumAlgorithm = checksumAlgorithm
        self.sizeBytes = sizeBytes
    }
}

struct ModelRuntimeVariant: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let backend: InferenceBackend
    let engineType: ASREngineType
    let runtimeLabel: String
    let platforms: [RuntimePlatform]
    let architectures: [String]
    let minimumOSVersion: String?
    let legacyModelId: String?
    let artifacts: [ModelArtifact]
    let isEnabled: Bool

    func isSupported(
        on platform: RuntimePlatform = .current,
        architecture: String = RuntimePlatform.currentArchitecture,
        osVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
    ) -> Bool {
        guard isEnabled else { return false }

        if !platforms.isEmpty, !platforms.contains(platform) {
            return false
        }

        if !architectures.isEmpty,
           !architectures.map({ $0.lowercased() }).contains(architecture.lowercased()) {
            return false
        }

        guard let minimumOSVersion, !minimumOSVersion.isEmpty else {
            return true
        }

        let versionParts = minimumOSVersion
            .split(separator: ".")
            .compactMap { Int($0) }
        guard !versionParts.isEmpty else { return true }

        let requiredMajor = versionParts[safe: 0] ?? 0
        let requiredMinor = versionParts[safe: 1] ?? 0
        let requiredPatch = versionParts[safe: 2] ?? 0

        let required = OperatingSystemVersion(
            majorVersion: requiredMajor,
            minorVersion: requiredMinor,
            patchVersion: requiredPatch
        )

        if osVersion.majorVersion != required.majorVersion {
            return osVersion.majorVersion > required.majorVersion
        }
        if osVersion.minorVersion != required.minorVersion {
            return osVersion.minorVersion >= required.minorVersion
        }
        return osVersion.patchVersion >= required.patchVersion
    }
}

struct ModelCard: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let displayName: String
    let parameterCount: String
    let sizeOnDisk: String
    let description: String
    let family: ModelFamily
    let languages: String
    let runtimeVariants: [ModelRuntimeVariant]

    var availableBackends: [InferenceBackend] {
        let unique = Array(Set(runtimeVariants.map(\.backend)))
        return unique.sorted(by: { $0.rawValue < $1.rawValue })
    }

    func preferredBackend(for platform: RuntimePlatform = .current) -> InferenceBackend {
        switch platform {
        case .ios:
            return availableBackends.contains(.cactus) ? .cactus : .legacy
        case .macOS, .catalyst:
            return availableBackends.contains(.mlx) ? .mlx : .legacy
        }
    }

    func variant(for variantId: String?) -> ModelRuntimeVariant? {
        guard let variantId else { return nil }
        return runtimeVariants.first(where: { $0.id == variantId })
    }
}

struct ModelCatalogManifest: Codable, Sendable {
    let schemaVersion: Int
    let cards: [ModelCard]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case cards
    }
}

enum ModelCatalogSource: String, Sendable {
    case remote
    case cached
    case bundled
    case legacy
}

struct ModelCatalog: Sendable {
    let source: ModelCatalogSource
    let cards: [ModelCard]
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}
