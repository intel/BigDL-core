#include "quantize.h"
#include <assert.h>
#include <math.h>
#include <cpuid.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define GGML_COMPUTE_FP16_TO_FP32(x) ((float)(x))
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define GGML_FP16_TO_FP32(x) ((float)(x))
#define GGML_FP32_TO_FP16(x) (x)

#else

static inline bool is_genuine_intel()
{
    unsigned int regs[4];
#ifdef _WIN32
    __cpuid((int *)regs, 0);
#else
    __get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[1] == 1970169159 && regs[2] == 1818588270 && regs[3] == 1231384169)
    {
        return true;
    }
    else
    {
        return false;
    }
}

static bool is_intel;

#if defined(_MSC_VER)
#define _mm256_dpbusd_epi32 _mm256_dpbusd_avx_epi32
#endif

#ifdef __F16C__

#ifdef _MSC_VER
#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#else
#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#endif

#elif defined(__POWER9_VECTOR__)

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
/* the inline asm below is about 12% faster than the lookup method */
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h)
{
    register float f;
    register double d;
    __asm__(
        "mtfprd %0,%2\n"
        "xscvhpdp %0,%0\n"
        "frsp %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */ "=f"(f) :
        /* in */ "r"(h));
    return f;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
    register double d;
    register ggml_fp16_t r;
    __asm__(/* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /* temp */ "=d"(d),
            /* out */ "=r"(r) :
            /* in */ "f"(f));
    return r;
}

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h)
{
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
                            (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000))
    {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#endif // __F16C__

#endif // __ARM_NEON

static void quantize_row_q4_0_gptq_reference(const float * RESTRICT in_src, int32_t * RESTRICT out_qweight, ggml_fp16_t * RESTRICT out_scale, int in_features) {
    static const int qk = QK4_0;

    assert(in_features % qk == 0);

    const int nb = in_features / qk;

    for (int i = 0; i < nb; i ++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = in_src[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        // Store the scale -> for each block
        out_scale[i] = GGML_FP32_TO_FP16(d);
        // out_scale[i] = GGML_COMPUTE_FP32_TO_FP16(d);

        for (int j = 0; j < qk; j += 8) {
            int index = i * qk + j;
#pragma unroll 8
            for (int z = 0; z < 8; z += 1) {
                const float x0 = in_src[index + z] * id;
                const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
                // For each of the 8 elements
                out_qweight[i * (qk / 8) + j / 8] |= (xi0 << (4 * z));
            }
        }
    }
}

size_t quantize_q4_0_to_qweight_and_scale(
    const float *src,
    int32_t *qweight,
    ggml_fp16_t *scale,
    int out_features,
    int in_features)
{
    // Quantize using QK4_0
    assert(in_features % QK4_0 == 0);

    const int nb = in_features / QK4_0;
    const int n = out_features * in_features;

    int64_t b = 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (b = 0; b < n; b += in_features)
    {
        // Quantize each row
        // 32bits / 4bits = 8
        int32_t *qweight_out = qweight + b / 8;
        ggml_fp16_t *scale_out = scale + b / QK4_0;
        quantize_row_q4_0_gptq_reference(src + b, qweight_out, scale_out, in_features);
    }

    return 0;
}