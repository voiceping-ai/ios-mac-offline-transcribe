import Foundation

public struct CactusRuntimeConfig: Sendable {
    public let modelDirectory: URL

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }
}

public final class CactusRuntime: @unchecked Sendable {
    private let config: CactusRuntimeConfig

    public init(config: CactusRuntimeConfig) {
        self.config = config
    }

    public func warmup() throws {
        guard FileManager.default.fileExists(atPath: config.modelDirectory.path) else {
            throw NSError(
                domain: "CactusRuntime",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model directory does not exist"]
            )
        }
    }

    public func transcribe(samples: [Float], language: String?) throws -> String {
        _ = language
        guard !samples.isEmpty else { return "" }
        throw NSError(
            domain: "CactusRuntime",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: "Cactus runtime bridge is not connected to the vendor SDK yet"]
        )
    }
}
