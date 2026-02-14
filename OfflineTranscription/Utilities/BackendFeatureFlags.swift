import Foundation

enum BackendFeatureFlags {
    private static let defaults = UserDefaults.standard

    static let cactusKey = "enableCactusBackend"
    static let mlxKey = "enableMLXBackend"
    static let selectorKey = "enableBackendSelectorUI"

    static var isBackendSelectorEnabled: Bool {
        boolValue(for: selectorKey, defaultValue: true)
    }

    static func isBackendEnabled(_ backend: InferenceBackend) -> Bool {
        switch backend {
        case .automatic, .legacy:
            return true
        case .cactus:
            return boolValue(for: cactusKey, defaultValue: defaultExperimentalBackendValue)
        case .mlx:
            return boolValue(for: mlxKey, defaultValue: defaultExperimentalBackendValue)
        }
    }

    private static var defaultExperimentalBackendValue: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    private static func boolValue(for key: String, defaultValue: Bool) -> Bool {
        if defaults.object(forKey: key) == nil {
            return defaultValue
        }
        return defaults.bool(forKey: key)
    }
}
