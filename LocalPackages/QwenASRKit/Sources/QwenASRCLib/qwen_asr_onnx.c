/*
 * qwen_asr_onnx.c - Qwen3-ASR ONNX Runtime inference pipeline
 *
 * Pipeline: audio → mel spectrogram → encoder ONNX → prompt embedding →
 *           decoder prefill ONNX → decode loop ONNX → token decode → text
 */

#include "qwen_asr_onnx.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include "qwen_asr.h"  /* for constants */
#include "ort/onnxruntime_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int qwen_onnx_verbose = 0;

/* ======================================================================== */
/* Constants                                                                 */
/* ======================================================================== */

#define MAX_DEC_LAYERS 28
#define MAX_NEW_TOKENS 1024
#define CHUNK_SIZE     100   /* mel frames per encoder chunk */

/* Prompt prefix: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|> */
static const int PROMPT_PREFIX[] = {151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669};
static const int N_PREFIX = 9;

/* Prompt suffix: <|audio_end|><|im_end|>\n<|im_start|>assistant\n */
static const int PROMPT_SUFFIX[] = {151670, 151645, 198, 151644, 77091, 198};
static const int N_SUFFIX = 6;

/* EOS tokens */
static const int EOS_TOKENS[] = {151643, 151645};
static const int N_EOS = 2;

/* ======================================================================== */
/* ONNX Context                                                              */
/* ======================================================================== */

struct qwen_onnx_ctx {
    const OrtApi    *api;
    OrtEnv          *env;
    OrtSession      *encoder;
    OrtSession      *prefill;
    OrtSession      *decode;
    OrtMemoryInfo   *mem_info;

    /* Token embeddings [vocab_size, hidden_dim] */
    float           *embed_tokens;
    int              vocab_size;
    int              hidden_dim;

    /* Decoder layer count */
    int              n_layers;

    /* Tokenizer */
    qwen_tokenizer_t *tokenizer;
};

/* ======================================================================== */
/* Helpers                                                                   */
/* ======================================================================== */

static int is_eos(int token) {
    for (int i = 0; i < N_EOS; i++)
        if (token == EOS_TOKENS[i]) return 1;
    return 0;
}

static int argmax_f32(const float *data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

/* Convert float16 (IEEE 754 half-precision) to float32 */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13);
    } else {
        f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* ======================================================================== */
/* NPY File Loader                                                           */
/* ======================================================================== */

/* Load a .npy file containing a 2D float32 or float16 array.
 * Always returns float32 data. Sets shape[0] and shape[1]. */
static float *load_npy(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "qwen_onnx: cannot open %s\n", path); return NULL; }

    /* Read magic + version */
    uint8_t header[10];
    if (fread(header, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(header, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }

    int major = header[6];
    uint32_t hdr_len;
    if (major == 1) {
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8);
    } else {
        /* v2: 4-byte header length at offset 8 */
        uint8_t extra[2];
        if (fread(extra, 1, 2, f) != 2) { fclose(f); return NULL; }
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8) |
                  ((uint32_t)extra[0] << 16) | ((uint32_t)extra[1] << 24);
    }

    char *hdr_str = (char *)malloc(hdr_len + 1);
    if (fread(hdr_str, 1, hdr_len, f) != hdr_len) { free(hdr_str); fclose(f); return NULL; }
    hdr_str[hdr_len] = '\0';

    /* Parse dtype: '<f4' (float32) or '<f2' (float16) */
    int is_fp16 = (strstr(hdr_str, "'<f2'") != NULL || strstr(hdr_str, "\"<f2\"") != NULL);

    /* Parse shape: (rows, cols) */
    char *sp = strstr(hdr_str, "shape");
    if (!sp) { free(hdr_str); fclose(f); return NULL; }
    char *lp = strchr(sp, '(');
    if (!lp) { free(hdr_str); fclose(f); return NULL; }
    int r = 0, c = 0;
    sscanf(lp + 1, "%d, %d", &r, &c);
    free(hdr_str);

    if (r <= 0 || c <= 0) { fclose(f); return NULL; }

    /* Read raw data */
    size_t n_elements = (size_t)r * c;
    float *data = (float *)malloc(n_elements * sizeof(float));

    if (is_fp16) {
        uint16_t *buf = (uint16_t *)malloc(n_elements * sizeof(uint16_t));
        if (fread(buf, sizeof(uint16_t), n_elements, f) != n_elements) {
            free(buf); free(data); fclose(f); return NULL;
        }
        for (size_t i = 0; i < n_elements; i++)
            data[i] = fp16_to_f32(buf[i]);
        free(buf);
    } else {
        if (fread(data, sizeof(float), n_elements, f) != n_elements) {
            free(data); fclose(f); return NULL;
        }
    }

    fclose(f);
    *rows = r;
    *cols = c;
    return data;
}

/* ======================================================================== */
/* ORT Helper Macros                                                         */
/* ======================================================================== */

#define ORT_CHECK(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *_m = ctx->api->GetErrorMessage(_s); \
        fprintf(stderr, "qwen_onnx ORT error: %s\n", _m); \
        ctx->api->ReleaseStatus(_s); \
        goto cleanup; \
    } \
} while(0)

#define ORT_CHECK_LOAD(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *_m = api->GetErrorMessage(_s); \
        fprintf(stderr, "qwen_onnx ORT error: %s\n", _m); \
        api->ReleaseStatus(_s); \
        qwen_onnx_free(ctx); \
        return NULL; \
    } \
} while(0)

/* ======================================================================== */
/* Load / Free                                                               */
/* ======================================================================== */

static char *path_join(const char *dir, const char *file) {
    size_t dlen = strlen(dir);
    size_t flen = strlen(file);
    char *p = (char *)malloc(dlen + flen + 2);
    memcpy(p, dir, dlen);
    if (dlen > 0 && dir[dlen-1] != '/') p[dlen++] = '/';
    memcpy(p + dlen, file, flen + 1);
    return p;
}

static const char *find_model(const char *dir, const char *base_name) {
    /* Try INT8 first, then full precision */
    static char buf[1024];
    /* Build int8 name: "encoder.onnx" → "encoder.int8.onnx" */
    const char *dot = strrchr(base_name, '.');
    if (dot) {
        size_t prefix_len = dot - base_name;
        snprintf(buf, sizeof(buf), "%s/%.*s.int8%s", dir, (int)prefix_len, base_name, dot);
        FILE *f = fopen(buf, "rb");
        if (f) { fclose(f); return buf; }
    }
    snprintf(buf, sizeof(buf), "%s/%s", dir, base_name);
    return buf;
}

qwen_onnx_ctx_t *qwen_onnx_load(const char *model_dir) {
    qwen_onnx_ctx_t *ctx = (qwen_onnx_ctx_t *)calloc(1, sizeof(qwen_onnx_ctx_t));
    if (!ctx) return NULL;

    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->api = api;

    /* Create ORT environment */
    ORT_CHECK_LOAD(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qwen_onnx", &ctx->env));
    ORT_CHECK_LOAD(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->mem_info));

    /* Session options — disable graph optimization for decoder (ORT optimizer bug) */
    OrtSessionOptions *enc_opts, *dec_opts;
    ORT_CHECK_LOAD(api->CreateSessionOptions(&enc_opts));
    ORT_CHECK_LOAD(api->SetSessionGraphOptimizationLevel(enc_opts, ORT_ENABLE_BASIC));
    ORT_CHECK_LOAD(api->SetIntraOpNumThreads(enc_opts, 4));

    ORT_CHECK_LOAD(api->CreateSessionOptions(&dec_opts));
    ORT_CHECK_LOAD(api->SetSessionGraphOptimizationLevel(dec_opts, ORT_DISABLE_ALL));
    ORT_CHECK_LOAD(api->SetIntraOpNumThreads(dec_opts, 4));

    /* Load encoder */
    const char *enc_path = find_model(model_dir, "encoder.onnx");
    if (qwen_onnx_verbose) fprintf(stderr, "Loading encoder: %s\n", enc_path);
    ORT_CHECK_LOAD(api->CreateSession(ctx->env, enc_path, enc_opts, &ctx->encoder));

    /* Load decoder prefill */
    const char *pf_path = find_model(model_dir, "decoder_prefill.onnx");
    if (qwen_onnx_verbose) fprintf(stderr, "Loading decoder prefill: %s\n", pf_path);
    ORT_CHECK_LOAD(api->CreateSession(ctx->env, pf_path, dec_opts, &ctx->prefill));

    /* Load decoder decode */
    const char *dc_path = find_model(model_dir, "decoder_decode.onnx");
    if (qwen_onnx_verbose) fprintf(stderr, "Loading decoder decode: %s\n", dc_path);
    ORT_CHECK_LOAD(api->CreateSession(ctx->env, dc_path, dec_opts, &ctx->decode));

    api->ReleaseSessionOptions(enc_opts);
    api->ReleaseSessionOptions(dec_opts);

    /* Load token embeddings */
    char *embed_path = path_join(model_dir, "embed_tokens.fp16.npy");
    FILE *ef = fopen(embed_path, "rb");
    if (!ef) {
        free(embed_path);
        embed_path = path_join(model_dir, "embed_tokens.npy");
    } else {
        fclose(ef);
    }

    int rows, cols;
    ctx->embed_tokens = load_npy(embed_path, &rows, &cols);
    free(embed_path);
    if (!ctx->embed_tokens) {
        fprintf(stderr, "qwen_onnx: failed to load embed_tokens\n");
        qwen_onnx_free(ctx);
        return NULL;
    }
    ctx->vocab_size = rows;
    ctx->hidden_dim = cols;
    if (qwen_onnx_verbose) fprintf(stderr, "Embeddings: %d x %d\n", rows, cols);

    /* Load tokenizer */
    char *vocab_path = path_join(model_dir, "vocab.json");
    ctx->tokenizer = qwen_tokenizer_load(vocab_path);
    free(vocab_path);
    if (!ctx->tokenizer) {
        fprintf(stderr, "qwen_onnx: failed to load tokenizer\n");
        qwen_onnx_free(ctx);
        return NULL;
    }

    /* Determine decoder layer count from number of prefill outputs.
     * Outputs: logits + n_layers K caches + n_layers V caches = 1 + 2*n_layers */
    size_t n_outputs;
    ORT_CHECK_LOAD(api->SessionGetOutputCount(ctx->prefill, &n_outputs));
    ctx->n_layers = (int)(n_outputs - 1) / 2;
    if (qwen_onnx_verbose) fprintf(stderr, "Decoder layers: %d\n", ctx->n_layers);

    return ctx;
}

void qwen_onnx_free(qwen_onnx_ctx_t *ctx) {
    if (!ctx) return;
    const OrtApi *api = ctx->api;
    if (api) {
        if (ctx->encoder)  api->ReleaseSession(ctx->encoder);
        if (ctx->prefill)  api->ReleaseSession(ctx->prefill);
        if (ctx->decode)   api->ReleaseSession(ctx->decode);
        if (ctx->mem_info) api->ReleaseMemoryInfo(ctx->mem_info);
        if (ctx->env)      api->ReleaseEnv(ctx->env);
    }
    if (ctx->embed_tokens) free(ctx->embed_tokens);
    if (ctx->tokenizer)    qwen_tokenizer_free(ctx->tokenizer);
    free(ctx);
}

/* ======================================================================== */
/* Transcription                                                             */
/* ======================================================================== */

char *qwen_onnx_transcribe(qwen_onnx_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return NULL;

    const OrtApi *api = ctx->api;
    const int n_layers = ctx->n_layers;
    const int hidden = ctx->hidden_dim;
    char *result = NULL;

    /* Tracking arrays for cleanup */
    OrtValue *enc_input = NULL, *enc_output = NULL;
    OrtValue *prefill_input = NULL;
    OrtValue **prefill_outputs = NULL;
    OrtValue *decode_token_input = NULL, *decode_pos_input = NULL;
    OrtValue **decode_outputs = NULL;
    OrtValue **kv_caches = NULL;  /* 2 * n_layers OrtValue pointers */
    float *input_embeds = NULL;
    int *generated = NULL;

    const int n_kv = 2 * n_layers;
    const int prefill_n_outputs = 1 + n_kv;
    const int decode_n_inputs = 2 + n_kv;
    const int decode_n_outputs = 1 + n_kv;

    prefill_outputs = (OrtValue **)calloc(prefill_n_outputs, sizeof(OrtValue *));
    decode_outputs  = (OrtValue **)calloc(decode_n_outputs, sizeof(OrtValue *));
    kv_caches       = (OrtValue **)calloc(n_kv, sizeof(OrtValue *));
    generated       = (int *)malloc(MAX_NEW_TOKENS * sizeof(int));
    if (!prefill_outputs || !decode_outputs || !kv_caches || !generated) goto cleanup;

    /* ---- Step 1: Mel spectrogram ---- */
    int n_frames;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &n_frames);
    if (!mel) { fprintf(stderr, "qwen_onnx: mel spectrogram failed\n"); goto cleanup; }

    /* Pad frames to multiple of CHUNK_SIZE */
    int pad_frames = (CHUNK_SIZE - (n_frames % CHUNK_SIZE)) % CHUNK_SIZE;
    int padded_frames = n_frames + pad_frames;
    if (pad_frames > 0) {
        float *padded = (float *)calloc((size_t)QWEN_MEL_BINS * padded_frames, sizeof(float));
        memcpy(padded, mel, (size_t)QWEN_MEL_BINS * n_frames * sizeof(float));
        free(mel);
        mel = padded;
    }
    if (qwen_onnx_verbose) fprintf(stderr, "Mel: %d x %d (padded from %d)\n", QWEN_MEL_BINS, padded_frames, n_frames);

    /* ---- Step 2: Run encoder ---- */
    {
        int64_t mel_shape[] = {1, QWEN_MEL_BINS, padded_frames};
        size_t mel_size = sizeof(float) * QWEN_MEL_BINS * padded_frames;
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, mel, mel_size,
                  mel_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_input));

        const char *enc_in_names[]  = {"mel_input"};
        const char *enc_out_names[] = {"audio_embeddings"};
        OrtValue *enc_inputs[]  = {enc_input};
        OrtValue *enc_outputs[] = {NULL};
        ORT_CHECK(api->Run(ctx->encoder, NULL, enc_in_names, (const OrtValue *const *)enc_inputs, 1,
                  enc_out_names, 1, enc_outputs));
        enc_output = enc_outputs[0];
    }
    free(mel); mel = NULL;

    /* Get audio embedding shape */
    OrtTensorTypeAndShapeInfo *enc_info;
    ORT_CHECK(api->GetTensorTypeAndShape(enc_output, &enc_info));
    int64_t enc_shape[3];
    ORT_CHECK(api->GetDimensions(enc_info, enc_shape, 3));
    api->ReleaseTensorTypeAndShapeInfo(enc_info);

    int n_audio = (int)enc_shape[1];
    if (qwen_onnx_verbose) fprintf(stderr, "Audio embeddings: %d tokens x %d dim\n", n_audio, (int)enc_shape[2]);

    float *audio_embeds;
    ORT_CHECK(api->GetTensorMutableData(enc_output, (void **)&audio_embeds));

    /* ---- Step 3: Build input embeddings ---- */
    int prompt_len = N_PREFIX + n_audio + N_SUFFIX;
    input_embeds = (float *)malloc((size_t)prompt_len * hidden * sizeof(float));
    if (!input_embeds) goto cleanup;

    /* Embed prefix tokens */
    for (int i = 0; i < N_PREFIX; i++)
        memcpy(input_embeds + i * hidden, ctx->embed_tokens + (size_t)PROMPT_PREFIX[i] * hidden, hidden * sizeof(float));

    /* Insert audio embeddings */
    memcpy(input_embeds + N_PREFIX * hidden, audio_embeds, (size_t)n_audio * hidden * sizeof(float));

    /* Embed suffix tokens */
    for (int i = 0; i < N_SUFFIX; i++)
        memcpy(input_embeds + (N_PREFIX + n_audio + i) * hidden,
               ctx->embed_tokens + (size_t)PROMPT_SUFFIX[i] * hidden, hidden * sizeof(float));

    /* ---- Step 4: Run decoder prefill ---- */
    {
        int64_t emb_shape[] = {1, prompt_len, hidden};
        size_t emb_size = sizeof(float) * prompt_len * hidden;
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, input_embeds, emb_size,
                  emb_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &prefill_input));

        /* Build output names */
        char output_name_bufs[1 + MAX_DEC_LAYERS * 2][24];
        const char *pf_out_names[1 + MAX_DEC_LAYERS * 2];
        pf_out_names[0] = "logits";
        for (int i = 0; i < n_layers; i++) {
            snprintf(output_name_bufs[1 + i], 24, "k_cache_%d", i);
            pf_out_names[1 + i] = output_name_bufs[1 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(output_name_bufs[1 + n_layers + i], 24, "v_cache_%d", i);
            pf_out_names[1 + n_layers + i] = output_name_bufs[1 + n_layers + i];
        }

        const char *pf_in_names[] = {"input_embeds"};
        OrtValue *pf_inputs[] = {prefill_input};
        ORT_CHECK(api->Run(ctx->prefill, NULL, pf_in_names, (const OrtValue *const *)pf_inputs, 1,
                  pf_out_names, prefill_n_outputs, prefill_outputs));
    }

    /* Extract first token from prefill logits */
    {
        float *logits;
        ORT_CHECK(api->GetTensorMutableData(prefill_outputs[0], (void **)&logits));
        int first_token = argmax_f32(logits, ctx->vocab_size);
        generated[0] = first_token;
        if (qwen_onnx_verbose) fprintf(stderr, "First token: %d\n", first_token);

        /* Transfer KV caches from prefill output */
        for (int i = 0; i < n_kv; i++) {
            kv_caches[i] = prefill_outputs[1 + i];
            prefill_outputs[1 + i] = NULL;  /* prevent double-free */
        }
        /* Free prefill logits */
        api->ReleaseValue(prefill_outputs[0]);
        prefill_outputs[0] = NULL;
    }

    /* ---- Step 5: Decode loop ---- */
    {
        int n_generated = 1;
        int token = generated[0];

        /* Pre-build input/output name strings */
        char in_name_bufs[2 + MAX_DEC_LAYERS * 2][24];
        const char *dc_in_names[2 + MAX_DEC_LAYERS * 2];
        dc_in_names[0] = "token_embed";
        dc_in_names[1] = "position";
        for (int i = 0; i < n_layers; i++) {
            snprintf(in_name_bufs[2 + i], 24, "k_cache_in_%d", i);
            dc_in_names[2 + i] = in_name_bufs[2 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(in_name_bufs[2 + n_layers + i], 24, "v_cache_in_%d", i);
            dc_in_names[2 + n_layers + i] = in_name_bufs[2 + n_layers + i];
        }

        char out_name_bufs[1 + MAX_DEC_LAYERS * 2][24];
        const char *dc_out_names[1 + MAX_DEC_LAYERS * 2];
        dc_out_names[0] = "logits";
        for (int i = 0; i < n_layers; i++) {
            snprintf(out_name_bufs[1 + i], 24, "k_cache_out_%d", i);
            dc_out_names[1 + i] = out_name_bufs[1 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(out_name_bufs[1 + n_layers + i], 24, "v_cache_out_%d", i);
            dc_out_names[1 + n_layers + i] = out_name_bufs[1 + n_layers + i];
        }

        for (int step = 0; step < MAX_NEW_TOKENS - 1; step++) {
            if (is_eos(token)) break;

            /* Create token embedding tensor */
            float *tok_emb = ctx->embed_tokens + (size_t)token * hidden;
            int64_t tok_shape[] = {1, 1, hidden};

            if (decode_token_input) { api->ReleaseValue(decode_token_input); decode_token_input = NULL; }
            ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, tok_emb,
                      hidden * sizeof(float), tok_shape, 3,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &decode_token_input));

            /* Create position tensor */
            int64_t pos_val = (int64_t)(prompt_len + step);
            int64_t pos_shape[] = {1};
            if (decode_pos_input) { api->ReleaseValue(decode_pos_input); decode_pos_input = NULL; }
            ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, &pos_val,
                      sizeof(int64_t), pos_shape, 1,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &decode_pos_input));

            /* Build input array: [token_embed, position, k_0..k_n, v_0..v_n] */
            OrtValue *dc_inputs[2 + MAX_DEC_LAYERS * 2];
            dc_inputs[0] = decode_token_input;
            dc_inputs[1] = decode_pos_input;
            for (int i = 0; i < n_kv; i++)
                dc_inputs[2 + i] = kv_caches[i];

            /* Run decode */
            memset(decode_outputs, 0, decode_n_outputs * sizeof(OrtValue *));
            ORT_CHECK(api->Run(ctx->decode, NULL, dc_in_names,
                      (const OrtValue *const *)dc_inputs, decode_n_inputs,
                      dc_out_names, decode_n_outputs, decode_outputs));

            /* Extract next token */
            float *logits;
            ORT_CHECK(api->GetTensorMutableData(decode_outputs[0], (void **)&logits));
            token = argmax_f32(logits, ctx->vocab_size);
            generated[n_generated++] = token;

            /* Free old KV caches, keep new ones */
            for (int i = 0; i < n_kv; i++) {
                api->ReleaseValue(kv_caches[i]);
                kv_caches[i] = decode_outputs[1 + i];
                decode_outputs[1 + i] = NULL;
            }
            /* Free decode logits */
            api->ReleaseValue(decode_outputs[0]);
            decode_outputs[0] = NULL;
        }

        if (qwen_onnx_verbose) fprintf(stderr, "Generated %d tokens\n", n_generated);

        /* Strip trailing EOS tokens */
        while (n_generated > 0 && is_eos(generated[n_generated - 1]))
            n_generated--;

        /* Decode tokens to text */
        /* Concatenate decoded token strings */
        size_t text_cap = 4096;
        char *text = (char *)malloc(text_cap);
        text[0] = '\0';
        size_t text_len = 0;

        int past_asr_text = 0;
        for (int i = 0; i < n_generated; i++) {
            if (generated[i] == QWEN_TOKEN_ASR_TEXT) {
                past_asr_text = 1;
                continue;
            }
            /* Skip language/special tokens before <asr_text> */
            if (!past_asr_text) continue;

            const char *piece = qwen_tokenizer_decode(ctx->tokenizer, generated[i]);
            if (piece) {
                size_t plen = strlen(piece);
                if (text_len + plen + 1 > text_cap) {
                    text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, piece, plen);
                text_len += plen;
                text[text_len] = '\0';
            }
        }

        /* If we never found <asr_text>, decode all tokens */
        if (!past_asr_text) {
            text_len = 0;
            text[0] = '\0';
            for (int i = 0; i < n_generated; i++) {
                /* Skip known special tokens */
                if (generated[i] >= 151643) continue;
                const char *piece = qwen_tokenizer_decode(ctx->tokenizer, generated[i]);
                if (piece) {
                    size_t plen = strlen(piece);
                    if (text_len + plen + 1 > text_cap) {
                        text_cap *= 2;
                        text = (char *)realloc(text, text_cap);
                    }
                    memcpy(text + text_len, piece, plen);
                    text_len += plen;
                    text[text_len] = '\0';
                }
            }
        }

        /* Trim leading/trailing whitespace */
        char *start = text;
        while (*start == ' ' || *start == '\n' || *start == '\t') start++;
        char *end = text + text_len;
        while (end > start && (end[-1] == ' ' || end[-1] == '\n' || end[-1] == '\t')) end--;

        size_t rlen = end - start;
        result = (char *)malloc(rlen + 1);
        memcpy(result, start, rlen);
        result[rlen] = '\0';
        free(text);
    }

cleanup:
    if (enc_input)  api->ReleaseValue(enc_input);
    if (enc_output) api->ReleaseValue(enc_output);
    if (prefill_input) api->ReleaseValue(prefill_input);
    if (decode_token_input) api->ReleaseValue(decode_token_input);
    if (decode_pos_input) api->ReleaseValue(decode_pos_input);
    if (prefill_outputs) {
        for (int i = 0; i < prefill_n_outputs; i++)
            if (prefill_outputs[i]) api->ReleaseValue(prefill_outputs[i]);
        free(prefill_outputs);
    }
    if (decode_outputs) {
        for (int i = 0; i < decode_n_outputs; i++)
            if (decode_outputs[i]) api->ReleaseValue(decode_outputs[i]);
        free(decode_outputs);
    }
    if (kv_caches) {
        for (int i = 0; i < n_kv; i++)
            if (kv_caches[i]) api->ReleaseValue(kv_caches[i]);
        free(kv_caches);
    }
    free(input_embeds);
    free(generated);
    return result;
}
