import Foundation

/// Darwin notification names for IPC between the app and the Broadcast Upload Extension.
enum DarwinNotifications {
    static let broadcastStarted: CFString = "com.voiceping.transcribe.broadcastStarted" as CFString
    static let broadcastStopped: CFString = "com.voiceping.transcribe.broadcastStopped" as CFString
    static let stopBroadcast = CFNotificationName("com.voiceping.transcribe.stopBroadcast" as CFString)
    static let audioReady: CFString = "com.voiceping.transcribe.audioReady" as CFString
}
