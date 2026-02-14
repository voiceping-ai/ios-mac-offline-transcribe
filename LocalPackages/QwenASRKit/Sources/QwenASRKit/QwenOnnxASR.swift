import Foundation
import QwenASRCLib

/// Thread-safe Swift wrapper around the Qwen3-ASR ONNX Runtime inference.
public final class QwenOnnxASR: @unchecked Sendable {
    private var ctx: UnsafeMutablePointer<qwen_onnx_ctx>?
    private let lock = NSLock()

    /// Load ONNX models from a directory containing encoder, decoder_prefill,
    /// decoder_decode ONNX models, embed_tokens.npy, and vocab.json.
    /// Returns nil if loading fails.
    public init?(modelDir: String) {
        qwen_onnx_verbose = 0
        guard let c = qwen_onnx_load(modelDir) else { return nil }
        self.ctx = c
    }

    deinit {
        lock.lock()
        if let c = ctx { qwen_onnx_free(c) }
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
            qwen_onnx_transcribe(c, buf.baseAddress, Int32(buf.count))
        }
        guard let result else { return nil }
        let text = String(cString: result)
        free(result)
        return text
    }

    /// Release all resources. Safe to call multiple times.
    public func release() {
        lock.lock()
        if let c = ctx { qwen_onnx_free(c) }
        ctx = nil
        lock.unlock()
    }
}
