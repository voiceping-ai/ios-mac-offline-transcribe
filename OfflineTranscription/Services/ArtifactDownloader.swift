import CryptoKit
import Foundation

struct ArtifactDownloadRequest: Sendable {
    let cardId: String
    let backend: InferenceBackend
    let version: String
    let artifacts: [ModelArtifact]
}

@MainActor
final class ArtifactDownloader {
    static var modelsRootDirectory: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("Models", isDirectory: true)
    }

    private let session: URLSession
    private let fileManager: FileManager

    private(set) var progress: Double = 0
    var onProgress: ((Double) -> Void)?

    init(
        session: URLSession = .shared,
        fileManager: FileManager = .default
    ) {
        self.session = session
        self.fileManager = fileManager
    }

    func localDirectory(for request: ArtifactDownloadRequest) -> URL {
        Self.modelsRootDirectory
            .appendingPathComponent(request.cardId, isDirectory: true)
            .appendingPathComponent(request.backend.rawValue, isDirectory: true)
            .appendingPathComponent(request.version, isDirectory: true)
    }

    func areArtifactsPresent(for request: ArtifactDownloadRequest) -> Bool {
        let destinationDirectory = localDirectory(for: request)
        return request.artifacts.allSatisfy { artifact in
            fileManager.fileExists(
                atPath: destinationDirectory
                    .appendingPathComponent(artifact.relativePath)
                    .path
            )
        }
    }

    func downloadArtifacts(for request: ArtifactDownloadRequest) async throws -> URL {
        let destinationDirectory = localDirectory(for: request)

        if areArtifactsPresent(for: request) {
            progress = 1
            onProgress?(1)
            return destinationDirectory
        }

        let stagingDirectory = fileManager.temporaryDirectory
            .appendingPathComponent("artifacts-\(UUID().uuidString)", isDirectory: true)

        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: stagingDirectory) }

        let total = Double(max(1, request.artifacts.count))
        for (index, artifact) in request.artifacts.enumerated() {
            let data = try await downloadData(from: artifact.url)
            try verifyChecksum(
                data: data,
                expectedChecksum: artifact.checksum,
                algorithm: artifact.checksumAlgorithm
            )

            let target = stagingDirectory.appendingPathComponent(artifact.relativePath)
            let targetDirectory = target.deletingLastPathComponent()
            try fileManager.createDirectory(at: targetDirectory, withIntermediateDirectories: true)
            try data.write(to: target, options: .atomic)

            let itemProgress = Double(index + 1) / total
            progress = itemProgress
            onProgress?(itemProgress)
        }

        try atomicallyReplaceDirectory(
            source: stagingDirectory,
            destination: destinationDirectory
        )
        try cleanupOldVersions(for: request, keeping: destinationDirectory)

        progress = 1
        onProgress?(1)

        return destinationDirectory
    }

    private func downloadData(from url: URL) async throws -> Data {
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NSError(
                domain: "ArtifactDownloader",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Artifact download failed for \(url.absoluteString)"]
            )
        }

        return data
    }

    private func verifyChecksum(
        data: Data,
        expectedChecksum: String,
        algorithm: ArtifactChecksumAlgorithm
    ) throws {
        switch algorithm {
        case .sha256:
            let digest = SHA256.hash(data: data)
            let actualChecksum = digest.map { String(format: "%02x", $0) }.joined()
            guard actualChecksum.caseInsensitiveCompare(expectedChecksum) == .orderedSame else {
                throw NSError(
                    domain: "ArtifactDownloader",
                    code: -2,
                    userInfo: [NSLocalizedDescriptionKey: "Checksum mismatch for downloaded artifact"]
                )
            }
        }
    }

    private func atomicallyReplaceDirectory(source: URL, destination: URL) throws {
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let backup = destination.deletingLastPathComponent()
            .appendingPathComponent(".old-\(UUID().uuidString)", isDirectory: true)

        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.moveItem(at: destination, to: backup)
        }

        do {
            try fileManager.moveItem(at: source, to: destination)
            if fileManager.fileExists(atPath: backup.path) {
                try fileManager.removeItem(at: backup)
            }
        } catch {
            if fileManager.fileExists(atPath: backup.path) {
                try? fileManager.moveItem(at: backup, to: destination)
            }
            throw error
        }
    }

    private func cleanupOldVersions(for request: ArtifactDownloadRequest, keeping keptDirectory: URL) throws {
        let backendDirectory = Self.modelsRootDirectory
            .appendingPathComponent(request.cardId, isDirectory: true)
            .appendingPathComponent(request.backend.rawValue, isDirectory: true)

        guard fileManager.fileExists(atPath: backendDirectory.path) else { return }

        let versionDirectories = try fileManager.contentsOfDirectory(
            at: backendDirectory,
            includingPropertiesForKeys: nil
        )

        for directory in versionDirectories where directory.path != keptDirectory.path {
            try? fileManager.removeItem(at: directory)
        }
    }
}
