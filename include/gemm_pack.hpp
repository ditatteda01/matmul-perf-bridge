#ifndef GEMM_PACK_HPP
#define GEMM_PACK_HPP

#include <cstddef>
#include <arm_neon.h>

void pack_tileA(
    const float32_t* __restrict A,
    float32_t* __restrict tileA,
    size_t stride_a, size_t mr, size_t kc, size_t MR
) {
    for (size_t j = 0; j < kc; j++) {
        for (size_t i = 0; i < MR; i++) {
            if (i < mr) {
                *tileA++ = A[i * stride_a + j];
            } else {
                *tileA++ = 0.0f;
            }
        }
    }
}

void pack_blockA(
    const float32_t* __restrict A,
    float32_t* __restrict blockA,
    size_t stride_a, size_t mc, size_t kc, size_t MR
) {
    for (size_t i = 0; i < mc; i += MR) {
        size_t mr = std::min(mc - i, MR);
        pack_tileA(A + i * stride_a, blockA, stride_a, mr, kc, MR);
        blockA += MR * kc;
    }
}

void pack_tileB(
    const float32_t* __restrict B,
    float32_t* __restrict tileB,
    size_t stride_b, size_t kc, size_t nr, size_t NR
) {
    for (size_t i = 0; i < kc; i++) {
        for (size_t j = 0; j < NR; j++) {
            if (j < nr) {
                *tileB++ = B[i * stride_b + j];
            } else {
                *tileB++ = 0.0f;
            }
        }
    }
}

void pack_blockB(
    const float32_t* __restrict B,
    float32_t* __restrict blockB,
    size_t stride_b, size_t kc, size_t nc, size_t NR) {
    for (size_t j = 0; j < nc; j += NR) {
        size_t nr = std::min(nc - j, NR);
        pack_tileB(B + j, blockB, stride_b, kc, nr, NR);
        blockB += kc * NR;
    }
}

void pack_tileC(
    float32_t* __restrict C,
    float32_t* __restrict tileC,
    size_t stride_c, size_t mr, size_t nr, size_t MR, size_t NR
) {
    std::fill(tileC, tileC + MR * NR, 0);
    for (size_t i = 0; i < mr; i++) {
        for (size_t j = 0; j < nr; j++) {
            tileC[i * NR + j] = C[i * stride_c + j];
        }
    }
}

void unpack_tileC(
    float32_t* __restrict C,
    float32_t* __restrict tileC,
    size_t stride_c, size_t mr, size_t nr, size_t NR) {
    for (size_t i = 0; i < mr; i++) {
        for (size_t j = 0; j < nr; j++) {
            C[i * stride_c + j] = tileC[i * NR + j];
        }
    }
}

#endif