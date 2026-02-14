import Foundation

public struct MLXRuntimeConfig: Sendable {
    public let modelDirectory: URL

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }
}

public final class MLXRuntime: @unchecked Sendable {
    private let config: MLXRuntimeConfig

    public init(config: MLXRuntimeConfig) {
        self.config = config
    }

    public func warmup() throws {
        guard FileManager.default.fileExists(atPath: config.modelDirectory.path) else {
            throw NSError(
                domain: "MLXRuntime",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model directory does not exist"]
            )
        }
    }

    public func transcribe(samples: [Float], language: String?) throws -> String {
        _ = language
        guard !samples.isEmpty else { return "" }
        throw NSError(
            domain: "MLXRuntime",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: "MLX runtime bridge is not connected to the model runner yet"]
        )
    }
}
