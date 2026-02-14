import Foundation
import QwenASRCLib

/// Thread-safe Swift wrapper around the qwen-asr C library.
public final class QwenASR: @unchecked Sendable {
    private var ctx: UnsafeMutablePointer<qwen_ctx_t>?
    private let lock = NSLock()

    /// Load a Qwen ASR model from a directory containing config.json, model.safetensors, vocab.json, merges.txt.
    /// Returns nil if model loading fails.
    public init?(modelDir: String) {
        let threads = Int32(Self.recommendedThreads())
        qwen_set_threads(threads)
        qwen_verbose = 0 // Suppress stderr logging on mobile
        guard let c = qwen_load(modelDir) else { return nil }
        // Configure for segmented offline mode (bounded memory)
        c.pointee.segment_sec = 20.0
        c.pointee.search_sec = 3.0
        self.ctx = c
    }

    deinit {
        lock.lock()
        if let c = ctx { qwen_free(c) }
        ctx = nil
        lock.unlock()
    }

    /// Transcribe Float32 audio samples (16kHz mono, range [-1, 1]).
    /// Returns transcribed text, or nil on failure.
    public func transcribe(samples: [Float]) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let c = ctx else { return nil }
        let result = samples.withUnsafeBufferPointer { buf in
            qwen_transcribe_audio(c, buf.baseAddress, Int32(buf.count))
        }
        guard let result else { return nil }
        let text = String(cString: result)
        free(result)
        return text
    }

    /// Set forced language (e.g. "English", "Japanese"). Pass nil for auto-detect.
    public func setLanguage(_ language: String?) {
        lock.lock()
        defer { lock.unlock() }
        guard let c = ctx else { return }
        if let lang = language {
            qwen_set_force_language(c, lang)
        } else {
            qwen_set_force_language(c, nil)
        }
    }

    /// Performance stats from last transcription.
    public var lastPerformance: (totalMs: Double, tokens: Int, audioMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        guard let c = ctx else { return (0, 0, 0) }
        return (c.pointee.perf_total_ms, Int(c.pointee.perf_text_tokens), c.pointee.perf_audio_ms)
    }

    /// Release all resources. Safe to call multiple times.
    public func release() {
        lock.lock()
        if let c = ctx { qwen_free(c) }
        ctx = nil
        lock.unlock()
    }

    private static func recommendedThreads() -> Int {
        let cores = max(ProcessInfo.processInfo.activeProcessorCount, 1)
        return min(cores / 2, 4)
    }
}
