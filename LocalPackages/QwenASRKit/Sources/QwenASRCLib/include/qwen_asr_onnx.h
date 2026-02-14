/*
 * qwen_asr_onnx.h - Qwen3-ASR ONNX Runtime Inference
 *
 * Runs Qwen3-ASR using ONNX Runtime. Requires encoder, decoder_prefill,
 * and decoder_decode ONNX models plus embed_tokens.npy.
 *
 * Links against onnxruntime symbols provided by SherpaOnnxKit.
 */

#ifndef QWEN_ASR_ONNX_H
#define QWEN_ASR_ONNX_H

#include <stddef.h>
#include <stdint.h>

typedef struct qwen_onnx_ctx qwen_onnx_ctx_t;

/* Load ONNX models from directory containing:
 * - encoder.int8.onnx (or encoder.onnx)
 * - decoder_prefill.int8.onnx (or decoder_prefill.onnx)
 * - decoder_decode.int8.onnx (or decoder_decode.onnx)
 * - embed_tokens.fp16.npy (or embed_tokens.npy)
 * - vocab.json
 * Returns NULL on error. */
qwen_onnx_ctx_t *qwen_onnx_load(const char *model_dir);

/* Free all resources. */
void qwen_onnx_free(qwen_onnx_ctx_t *ctx);

/* Transcribe raw audio (mono float32, 16kHz).
 * Returns allocated string (caller must free), or NULL on error. */
char *qwen_onnx_transcribe(qwen_onnx_ctx_t *ctx, const float *samples, int n_samples);

/* Global verbose flag (shared with qwen_verbose) */
extern int qwen_onnx_verbose;

#endif /* QWEN_ASR_ONNX_H */
