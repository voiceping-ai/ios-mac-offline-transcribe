import XCTest
@testable import OfflineTranscription

@MainActor
final class ModelCatalogServiceTests: XCTestCase {

    func testLocalFallbackCatalogAlwaysReturnsCards() {
        let service = ModelCatalogService()
        let catalog = service.loadLocalFallbackCatalog()
        XCTAssertFalse(catalog.cards.isEmpty)
    }

    func testCatalogContainsLegacyCardsForKnownModel() {
        let service = ModelCatalogService()
        let catalog = service.loadLocalFallbackCatalog()
        let whisperBaseCard = catalog.cards.first(where: { $0.id == "whisper-base" })
        XCTAssertNotNil(whisperBaseCard)

        let hasLegacyVariant = whisperBaseCard?.runtimeVariants.contains(where: {
            $0.backend == .legacy
        }) ?? false
        XCTAssertTrue(hasLegacyVariant)
    }

    func testRuntimeVariantPlatformGate() {
        let variant = ModelRuntimeVariant(
            id: "platform-gated",
            backend: .legacy,
            engineType: .whisperKit,
            runtimeLabel: "CoreML",
            platforms: [.ios],
            architectures: [],
            minimumOSVersion: nil,
            legacyModelId: "whisper-base",
            artifacts: [],
            isEnabled: true
        )

        XCTAssertTrue(variant.isSupported(on: .ios))
        XCTAssertFalse(variant.isSupported(on: .macOS))
    }
}
