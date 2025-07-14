#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <immintrin.h>

#define QK4_0 64

#ifdef GGML_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef GGML_BUILD
#define GGML_API __declspec(dllexport)
#else
#define GGML_API __declspec(dllimport)
#endif
#else
#define GGML_API __attribute__((visibility("default")))
#endif
#else
#define GGML_API
#endif

#ifdef  __cplusplus
extern "C" {
#endif
#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_fp16_t;
#else
    typedef uint16_t ggml_fp16_t;
#endif

#ifdef __cplusplus
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

GGML_API size_t quantize_q4_0_to_qweight_and_scale(
    const float *src,
    int32_t *qweight,
    ggml_fp16_t *scale,
    int out_features,
    int in_features);

#ifdef __cplusplus
}
#endif