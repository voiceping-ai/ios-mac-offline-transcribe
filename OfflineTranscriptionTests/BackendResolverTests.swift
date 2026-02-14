import XCTest
@testable import OfflineTranscription

@MainActor
final class BackendResolverTests: XCTestCase {

    override func setUp() {
        super.setUp()
        UserDefaults.standard.set(true, forKey: BackendFeatureFlags.cactusKey)
        UserDefaults.standard.set(true, forKey: BackendFeatureFlags.mlxKey)
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: BackendFeatureFlags.cactusKey)
        UserDefaults.standard.removeObject(forKey: BackendFeatureFlags.mlxKey)
        super.tearDown()
    }

    func testIOSPrefersCactusWhenAvailable() {
        let card = makeDualBackendCard()
        let result = BackendResolver.shared.resolve(
            card: card,
            requestedBackend: .automatic,
            platform: .ios
        )

        XCTAssertEqual(result.effectiveBackend, .cactus)
        XCTAssertEqual(result.runtimeVariant?.engineType, .cactus)
    }

    func testMacPrefersMLXWhenAvailable() {
        let card = makeDualBackendCard()
        let result = BackendResolver.shared.resolve(
            card: card,
            requestedBackend: .automatic,
            platform: .macOS
        )

        XCTAssertEqual(result.effectiveBackend, .mlx)
        XCTAssertEqual(result.runtimeVariant?.engineType, .mlx)
    }

    func testFallbackToLegacyWhenRequestedBackendUnavailable() {
        let card = ModelCard(
            id: "legacy-only",
            displayName: "Legacy Only",
            parameterCount: "10M",
            sizeOnDisk: "~10 MB",
            description: "test",
            family: .whisper,
            languages: "en",
            runtimeVariants: [
                ModelRuntimeVariant(
                    id: "legacy-only-v1",
                    backend: .legacy,
                    engineType: .whisperKit,
                    runtimeLabel: "CoreML (WhisperKit)",
                    platforms: [.ios, .macOS],
                    architectures: [],
                    minimumOSVersion: nil,
                    legacyModelId: "whisper-base",
                    artifacts: [],
                    isEnabled: true
                )
            ]
        )

        let result = BackendResolver.shared.resolve(
            card: card,
            requestedBackend: .cactus,
            platform: .ios
        )

        XCTAssertEqual(result.effectiveBackend, .legacy)
        XCTAssertEqual(result.runtimeVariant?.engineType, .whisperKit)
        XCTAssertNotNil(result.fallbackReason)
    }

    private func makeDualBackendCard() -> ModelCard {
        ModelCard(
            id: "dual-backend",
            displayName: "Dual",
            parameterCount: "50M",
            sizeOnDisk: "~100 MB",
            description: "test",
            family: .whisper,
            languages: "en",
            runtimeVariants: [
                ModelRuntimeVariant(
                    id: "legacy",
                    backend: .legacy,
                    engineType: .whisperKit,
                    runtimeLabel: "CoreML (WhisperKit)",
                    platforms: [.ios, .macOS, .catalyst],
                    architectures: [],
                    minimumOSVersion: nil,
                    legacyModelId: "whisper-base",
                    artifacts: [],
                    isEnabled: true
                ),
                ModelRuntimeVariant(
                    id: "cactus",
                    backend: .cactus,
                    engineType: .cactus,
                    runtimeLabel: "Cactus Runtime",
                    platforms: [.ios],
                    architectures: [],
                    minimumOSVersion: nil,
                    legacyModelId: nil,
                    artifacts: [],
                    isEnabled: true
                ),
                ModelRuntimeVariant(
                    id: "mlx",
                    backend: .mlx,
                    engineType: .mlx,
                    runtimeLabel: "MLX Runtime",
                    platforms: [.macOS],
                    architectures: [],
                    minimumOSVersion: nil,
                    legacyModelId: nil,
                    artifacts: [],
                    isEnabled: true
                )
            ]
        )
    }
}
