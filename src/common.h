#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef D_FEAT
#define D_FEAT 256      /* features per sample (1 sample = 1 KB) */
#endif
#ifndef H_HID
#define H_HID 128       /* hidden neurons */
#endif
#ifndef BLOCK_SAMPLES
#define BLOCK_SAMPLES 16384   /* 16 MB working buffer per worker */
#endif

/* AI workload: 1-hidden-layer neural network inference on each sample.
   h = ReLU(W x + b), out = v . h.  Result = sum of all outputs (checksum). */
static float W[H_HID][D_FEAT];
static float B[H_HID];
static float V[H_HID];

static inline float hash_float(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (float)(x >> 40) / 8388608.0f - 1.0f;
}

static void init_weights(void)
{
    for (int j = 0; j < H_HID; j++) {
        for (int i = 0; i < D_FEAT; i++)
            W[j][i] = hash_float(1000000000ULL + (uint64_t)j * D_FEAT + i) * 0.1f;
        B[j] = hash_float(2000000000ULL + j) * 0.1f;
        V[j] = hash_float(3000000000ULL + j) * 0.1f;
    }
}

static void fill_samples(float *buf, long first, long count)
{
    for (long s = 0; s < count; s++)
        for (int k = 0; k < D_FEAT; k++)
            buf[s * D_FEAT + k] = hash_float((uint64_t)(first + s) * D_FEAT + k);
}

static double run_network(const float *data, long count)
{
    double total = 0.0;
    for (long s = 0; s < count; s++) {
        const float *x = data + s * D_FEAT;
        float out = 0.0f;
        for (int j = 0; j < H_HID; j++) {
            float acc = B[j];
            for (int i = 0; i < D_FEAT; i++) acc += W[j][i] * x[i];
            if (acc > 0.0f) out += V[j] * acc;
        }
        total += out;
    }
    return total;
}

/* streaming: create + process samples one block at a time */
static double process_range(float *buf, long first, long count)
{
    double total = 0.0;
    for (long done = 0; done < count; done += BLOCK_SAMPLES) {
        long m = count - done < BLOCK_SAMPLES ? count - done : BLOCK_SAMPLES;
        fill_samples(buf, first + done, m);
        total += run_network(buf, m);
    }
    return total;
}

static long samples_for_gb(double gb)
{
    return (long)(gb * 1024.0 * 1024.0 * 1024.0 / (D_FEAT * sizeof(float)));
}
#endif
