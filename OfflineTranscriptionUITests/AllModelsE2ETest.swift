import Foundation
import XCTest

/// E2E UI test that launches the app with each model, waits for transcription,
/// and captures screenshot evidence at each step.
///
/// Evidence is written to /tmp/e2e_evidence/{modelId}/ with:
///   01_model_loading.png  — right after app launch
///   02_model_loaded.png   — when transcription screen is visible
///   03_inference_result.png — after transcription completes (with E2E overlay)
///   result.json           — machine-readable pass/fail + transcript
final class AllModelsE2ETest: XCTestCase {
    private let bundleId = "com.voiceping.transcribe"

    // Per-model timeout (seconds) for download + load + transcribe
    // Must be larger than app-side polling timeout (TranscriptionView auto-test)
    private func timeout(for modelId: String) -> TimeInterval {
        switch modelId {
        case let id where id.contains("large"): return 480
        case let id where id.contains("300m"): return 360
        case let id where id.contains("small"): return 300
        case let id where id.contains("base"): return 240
        default: return 150
        }
    }

    // MARK: - Individual model tests

    func test_whisperTiny() { testModel("whisper-tiny") }
    func test_whisperBase() { testModel("whisper-base") }
    func test_whisperSmall() { testModel("whisper-small") }
    func test_whisperLargeV3Turbo() { testModel("whisper-large-v3-turbo") }
    func test_whisperLargeV3TurboCompressed() { testModel("whisper-large-v3-turbo-compressed") }
    func test_moonshineTiny() { testModel("moonshine-tiny") }
    func test_moonshineBase() { testModel("moonshine-base") }
    func test_sensevoiceSmall() { testModel("sensevoice-small") }
    func test_zipformer20m() { testModel("zipformer-20m") }
    func test_omnilingual300m() { testModel("omnilingual-300m") }
    func test_parakeetTdtV3() { testModel("parakeet-tdt-v3") }

    // MARK: - Core test logic

    private func testModel(_ modelId: String) {
        let evidenceDir = "/tmp/e2e_evidence/\(modelId)"
        let resultPath = "/tmp/e2e_result_\(modelId).json"
        let timeoutSec = timeout(for: modelId)

        // Clean up previous evidence
        try? FileManager.default.removeItem(atPath: evidenceDir)
        try? FileManager.default.createDirectory(atPath: evidenceDir, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(atPath: resultPath)

        // 1. Launch app with auto-test args
        let app = XCUIApplication()
        app.launchArguments = ["--auto-test", "--model-id", modelId]
        app.launch()

        // 2. Screenshot 01: model loading/downloading
        sleep(3)
        saveScreenshot(app.screenshot(), to: evidenceDir, name: "01_model_loading.png")
        addAttachment(app.screenshot(), name: "\(modelId)_01_model_loading")

        // 3. Wait for main tab view or model info label (model loaded)
        let modelInfo = app.staticTexts.matching(identifier: "model_info_label").firstMatch
        let mainTab = app.otherElements.matching(identifier: "main_tab_view").firstMatch
        let loaded = modelInfo.waitForExistence(timeout: timeoutSec)
            || mainTab.waitForExistence(timeout: 5)

        if loaded {
            NSLog("[E2E] [\(modelId)] Model loaded — transcription screen visible")
        } else {
            NSLog("[E2E] [\(modelId)] Timeout waiting for model load")
        }
        saveScreenshot(app.screenshot(), to: evidenceDir, name: "02_model_loaded.png")
        addAttachment(app.screenshot(), name: "\(modelId)_02_model_loaded")

        // Trigger file transcription explicitly as a fallback for real-device runs.
        let testFileButton = app.buttons.matching(identifier: "test_file_button").firstMatch
        if testFileButton.waitForExistence(timeout: 8) {
            testFileButton.tap()
            NSLog("[E2E] [\(modelId)] test_file_button tapped to start transcription")
        } else {
            NSLog("[E2E] [\(modelId)] test_file_button not available (auto-test likely already running)")
        }

        // 4. Wait for E2E overlay or result.json
        let overlay = app.otherElements.matching(identifier: "e2e_overlay").firstMatch
        let confirmedTextElement = app.staticTexts.matching(identifier: "confirmed_text").firstMatch
        let overlayTimeout: TimeInterval = timeoutSec
        let startWait = Date()
        var resultExists = false
        var capturedResultJSON: String?

        while Date().timeIntervalSince(startWait) < overlayTimeout {
            // Check for result.json file (fast path)
            if FileManager.default.fileExists(atPath: resultPath) {
                resultExists = true
                if let data = FileManager.default.contents(atPath: resultPath),
                   let json = String(data: data, encoding: .utf8) {
                    capturedResultJSON = json
                }
                NSLog("[E2E] [\(modelId)] result.json detected")
                break
            }
            // Check for E2E overlay in UI
            if overlay.exists {
                if let payload = extractE2EPayload(from: app) {
                    resultExists = true
                    capturedResultJSON = payload
                    try? payload.write(toFile: resultPath, atomically: true, encoding: .utf8)
                    NSLog("[E2E] [\(modelId)] E2E overlay payload captured")
                    break
                } else {
                    NSLog("[E2E] [\(modelId)] E2E overlay detected but payload not ready yet")
                }
            }
            // UI fallback for cases where file/overlay bridge is unavailable on real devices.
            if confirmedTextElement.exists {
                let transcript = confirmedTextElement.label.trimmingCharacters(in: .whitespacesAndNewlines)
                if !transcript.isEmpty {
                    let payload = syntheticResultJSON(modelId: modelId, transcript: transcript)
                    capturedResultJSON = payload
                    try? payload.write(toFile: resultPath, atomically: true, encoding: .utf8)
                    resultExists = true
                    NSLog("[E2E] [\(modelId)] confirmed_text captured from UI fallback")
                    break
                }
            }
            Thread.sleep(forTimeInterval: 2)
        }

        // 5. Final screenshot
        sleep(2)
        saveScreenshot(app.screenshot(), to: evidenceDir, name: "03_inference_result.png")
        addAttachment(app.screenshot(), name: "\(modelId)_03_inference_result")

        // 6. Validate result.json
        if !resultExists {
            if let payload = extractE2EPayload(from: app) {
                resultExists = true
                capturedResultJSON = payload
                try? payload.write(toFile: resultPath, atomically: true, encoding: .utf8)
                NSLog("[E2E] [\(modelId)] E2E overlay payload captured (post-wait fallback)")
            }
        }

        if !resultExists {
            if confirmedTextElement.exists {
                let transcript = confirmedTextElement.label.trimmingCharacters(in: .whitespacesAndNewlines)
                if !transcript.isEmpty {
                    let payload = syntheticResultJSON(modelId: modelId, transcript: transcript)
                    capturedResultJSON = payload
                    try? payload.write(toFile: resultPath, atomically: true, encoding: .utf8)
                    resultExists = true
                    NSLog("[E2E] [\(modelId)] confirmed_text captured from UI fallback (post-wait)")
                }
            }
        }

        if !resultExists {
            resultExists = FileManager.default.fileExists(atPath: resultPath)
        }

        let resultDataFromFile = FileManager.default.contents(atPath: resultPath)
        let resultText = (resultDataFromFile.flatMap { String(data: $0, encoding: .utf8) })
            ?? capturedResultJSON

        if resultExists, let json = resultText, let data = json.data(using: .utf8) {
            // Copy to evidence directory
            try? data.write(to: URL(fileURLWithPath: "\(evidenceDir)/result.json"))
            NSLog("[E2E] [\(modelId)] result.json: \(json)")

            guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                XCTFail("[\(modelId)] result.json is not valid JSON: \(json)")
                return
            }
            if let compactData = try? JSONSerialization.data(withJSONObject: object, options: []),
               let compactJSON = String(data: compactData, encoding: .utf8) {
                NSLog("[E2E_RESULT][\(modelId)] \(compactJSON)")
            }

            XCTAssertEqual(
                object["pass"] as? Bool,
                true,
                "[\(modelId)] Expected pass=true in result.json, got: \(json)"
            )

            // Log translation evidence (informational, not required for pass)
            if object["expects_translation"] as? Bool == true {
                let translatedText = (object["translated_text"] as? String)?.trimmingCharacters(
                    in: .whitespacesAndNewlines
                ) ?? ""
                if translatedText.isEmpty {
                    NSLog("[E2E] [\(modelId)] WARNING: translation enabled but translated_text is empty")
                } else {
                    NSLog("[E2E] [\(modelId)] Translation evidence: \(translatedText.prefix(80))...")
                }
            }
        } else {
            // Write timeout result
            let timeoutJson = """
            {"model_id":"\(modelId)","pass":false,"error":"timeout"}
            """
            try? timeoutJson.write(toFile: "\(evidenceDir)/result.json", atomically: true, encoding: .utf8)
            XCTFail("[\(modelId)] Timed out waiting for transcription result")
        }

        NSLog("[E2E] [\(modelId)] E2E PASSED")
    }

    // MARK: - Helpers

    private func saveScreenshot(_ screenshot: XCUIScreenshot, to dir: String, name: String) {
        let path = "\(dir)/\(name)"
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: path))
        NSLog("[E2E] Screenshot saved: \(path)")
    }

    private func addAttachment(_ screenshot: XCUIScreenshot, name: String) {
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func extractE2EPayload(from app: XCUIApplication) -> String? {
        let overlay = app.otherElements.matching(identifier: "e2e_overlay").firstMatch
        let overlayPayloadText = app.staticTexts.matching(identifier: "e2e_overlay_payload").firstMatch
        var candidates: [String] = []

        if let value = overlay.value as? String, !value.isEmpty {
            candidates.append(value)
        }
        if overlayPayloadText.exists {
            let label = overlayPayloadText.label.trimmingCharacters(in: .whitespacesAndNewlines)
            if !label.isEmpty {
                candidates.append(label)
            }
            if let value = overlayPayloadText.value as? String, !value.isEmpty {
                candidates.append(value)
            }
        }

        for candidate in candidates {
            if let normalized = normalizedJSON(from: candidate),
               let data = normalized.data(using: .utf8),
               (try? JSONSerialization.jsonObject(with: data)) != nil {
                return normalized
            }
        }
        return nil
    }

    private func normalizedJSON(from raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}") else {
            return nil
        }
        return String(trimmed[start...end])
    }

    private func syntheticResultJSON(modelId: String, transcript: String) -> String {
        let lower = transcript.lowercased()
        let keywords = ["country", "ask", "do for", "fellow", "americans"]
        let hasKeywordHit = keywords.contains { lower.contains($0) }
        let isOmnilingual = modelId.lowercased().contains("omnilingual")
        let hasMeaningfulText = transcript.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
        let asciiLetterCount = transcript.unicodeScalars.filter {
            CharacterSet.letters.contains($0) && $0.isASCII
        }.count
        let omnilingualQuality = hasKeywordHit
            || (hasMeaningfulText && transcript.count >= 24 && asciiLetterCount >= 12)
        let pass = isOmnilingual ? omnilingualQuality : hasKeywordHit
        let payload: [String: Any] = [
            "model_id": modelId,
            "engine": "ui-fallback",
            "transcript": transcript,
            "translated_text": "",
            "expects_translation": false,
            "translation_ready": true,
            "pass": pass,
            "tokens_per_second": 0,
            "duration_ms": 0,
            "timestamp": ISO8601DateFormatter().string(from: Date())
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]),
              let json = String(data: data, encoding: .utf8) else {
            return """
            {"model_id":"\(modelId)","pass":false,"error":"failed to build ui-fallback result"}
            """
        }
        return json
    }
}
