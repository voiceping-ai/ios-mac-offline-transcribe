import Foundation
import whisper

public struct CactusRuntimeConfig: Sendable {
    public let modelDirectory: URL

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }
}

public struct CactusTranscriptionResult: Sendable {
    public let text: String
    public let language: String?

    public struct Segment: Sendable {
        public let text: String
        public let startMs: Int64
        public let endMs: Int64
    }

    public let segments: [Segment]
}

public final class CactusRuntime: @unchecked Sendable {
    private let config: CactusRuntimeConfig
    private var context: OpaquePointer?

    public init(config: CactusRuntimeConfig) {
        self.config = config
    }

    deinit {
        if let context {
            whisper_free(context)
        }
    }

    public func warmup() throws {
        let modelPath = try findModelFile()

        var params = whisper_context_default_params()
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #else
        params.use_gpu = true
        #endif

        guard let ctx = whisper_init_from_file_with_params(modelPath, params) else {
            throw NSError(
                domain: "CactusRuntime",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize whisper.cpp context from \(modelPath)"]
            )
        }

        self.context = ctx
    }

    public func transcribe(samples: [Float], language: String?) throws -> CactusTranscriptionResult {
        guard let context else {
            throw NSError(
                domain: "CactusRuntime",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Whisper context not initialized. Call warmup() first."]
            )
        }

        guard !samples.isEmpty else {
            return CactusTranscriptionResult(text: "", language: language, segments: [])
        }

        let threadCount = max(1, ProcessInfo.processInfo.activeProcessorCount - 2)

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads = Int32(threadCount)
        params.print_progress = false
        params.print_special = false
        params.print_realtime = false
        params.print_timestamps = false
        params.single_segment = false
        params.no_timestamps = false

        let langCStr = strdup((language ?? "auto") as NSString as String)
        defer { free(langCStr) }
        params.language = UnsafePointer(langCStr)

        let result = samples.withUnsafeBufferPointer { buffer in
            whisper_full(context, params, buffer.baseAddress, Int32(buffer.count))
        }

        guard result == 0 else {
            throw NSError(
                domain: "CactusRuntime",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "whisper_full() failed with code \(result)"]
            )
        }

        let nSegments = whisper_full_n_segments(context)
        var segments: [CactusTranscriptionResult.Segment] = []
        var fullText = ""

        for i in 0..<nSegments {
            guard let cText = whisper_full_get_segment_text(context, i) else { continue }
            let segmentText = String(cString: cText)
            let t0 = whisper_full_get_segment_t0(context, i)
            let t1 = whisper_full_get_segment_t1(context, i)

            segments.append(CactusTranscriptionResult.Segment(
                text: segmentText,
                startMs: Int64(t0) * 10,
                endMs: Int64(t1) * 10
            ))
            fullText += segmentText
        }

        let detectedLang: String?
        let langId = whisper_full_lang_id(context)
        if langId >= 0 {
            if let cLang = whisper_lang_str(langId) {
                detectedLang = String(cString: cLang)
            } else {
                detectedLang = language
            }
        } else {
            detectedLang = language
        }

        return CactusTranscriptionResult(
            text: fullText.trimmingCharacters(in: .whitespacesAndNewlines),
            language: detectedLang,
            segments: segments
        )
    }

    private func findModelFile() throws -> String {
        let dir = config.modelDirectory
        let fm = FileManager.default

        guard fm.fileExists(atPath: dir.path) else {
            throw NSError(
                domain: "CactusRuntime",
                code: -4,
                userInfo: [NSLocalizedDescriptionKey: "Model directory does not exist: \(dir.path)"]
            )
        }

        let contents = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        if let binFile = contents.first(where: { $0.pathExtension == "bin" }) {
            return binFile.path
        }

        throw NSError(
            domain: "CactusRuntime",
            code: -5,
            userInfo: [NSLocalizedDescriptionKey: "No .bin model file found in \(dir.path)"]
        )
    }
}
