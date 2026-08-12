#include "env_test.hpp"
#include "matmul.hpp"

namespace env_test {
    std::string greet(const std::string& name) {
        return "Hello, " + name + " from cplusplus module!";
    }
}

namespace matmul {
    template <typename T>
    void matmul_rMajor_vector(
        const std::vector< std::vector<T> >& A,
        const std::vector< std::vector<T> >& B,
        std::vector< std::vector<T> >& C,
        size_t M, size_t K, size_t N) {
            for (size_t i = 0; i < M; i++) {
                for (size_t k = 0; k < K; k++) {
                    for (size_t j = 0; j < N; j++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
    }

    template <typename T>
    void matmul_cMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t M, size_t K, size_t N) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }

    template <typename T>
    void matmul_rMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t M, size_t K, size_t N) {
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                T valA = A[i * K + k];
                for (size_t j = 0; j < N; j++) {
                    C[i * N + j] += valA * B[k * N + j];
                }
            }
        }
    }

    // Force Compiler to generate compiled binary code so that dlopen can find the symbols at runtime.
    template void matmul_rMajor_vector(const std::vector< std::vector<int32_t> >& A, const std::vector< std::vector<int32_t> >& B, std::vector< std::vector<int32_t> >& C, size_t M, size_t K, size_t N);
    template void matmul_rMajor_vector(const std::vector< std::vector<float32_t> >& A, const std::vector< std::vector<float32_t> >& B, std::vector< std::vector<float32_t> >& C, size_t M, size_t K, size_t N);
    template void matmul_cMajor_ptr<float32_t>(const float32_t*, const float32_t*, float32_t*, size_t, size_t, size_t);
    template void matmul_cMajor_ptr<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t);
    template void matmul_rMajor_ptr<float32_t>(const float32_t*, const float32_t*, float32_t*, size_t, size_t, size_t);
    template void matmul_rMajor_ptr<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t);

}

namespace gemm_f32_kernel4x16 {
    void pack_tileA(const float32_t* __restrict A, float32_t* __restrict tileA, size_t stride_a, size_t mr, size_t kc) {
        if (mr == 4) {
            for (size_t j = 0; j < kc; j++) {
                tileA[0] = A[j];
                tileA[1] = A[stride_a + j];
                tileA[2] = A[2 * stride_a + j];
                tileA[3] = A[3 * stride_a + j];
                tileA += 4;
            }
        } else {
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
    }

    void pack_blockA(const float32_t* __restrict A, float32_t* __restrict blockA, size_t stride_a, size_t mc, size_t kc) {
        for (size_t i = 0; i < mc; i += MR) {
            size_t mr = std::min(mc - i, MR);
            pack_tileA(A + i * stride_a, blockA, stride_a, mr, kc);
            blockA += MR * kc;
        }
    }

    void pack_tileB(const float32_t* __restrict B, float32_t* __restrict tileB, size_t stride_b, size_t kc, size_t nr) {
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

    void pack_blockB(const float32_t* __restrict B, float32_t* __restrict blockB, size_t stride_b, size_t kc, size_t nc) {
        for (size_t j = 0; j < nc; j += NR) {
            size_t nr = std::min(nc - j, NR);
            pack_tileB(B + j, blockB, stride_b, kc, nr);
            blockB += kc * NR;
        }
    }

    void pack_tileC(float32_t* __restrict C, float32_t* __restrict tileC, size_t stride_c, size_t mr, size_t nr) {
        std::fill(tileC, tileC + MR * NR, 0);
        for (size_t i = 0; i < mr; i++) {
            for (size_t j = 0; j < nr; j++) {
                tileC[i * NR + j] = C[i * stride_c + j];
            }
        }
    }

    void unpack_tileC(float32_t* __restrict C, float32_t* __restrict tileC, size_t stride_c, size_t mr, size_t nr) {
        for (size_t i = 0; i < mr; i++) {
            for (size_t j = 0; j < nr; j++) {
                C[i * stride_c + j] = tileC[i * NR + j];
            }
        }
    }

    void kernel_fma_4x16_edge(float32_t* __restrict tileA_packed, float32_t* __restrict tileB_packed, float32_t* __restrict tileC_packed, size_t kc) {
        float32x4_t c00 = vld1q_f32(tileC_packed);
        float32x4_t c01 = vld1q_f32(tileC_packed + 4);
        float32x4_t c02 = vld1q_f32(tileC_packed + 8);
        float32x4_t c03 = vld1q_f32(tileC_packed + 12);
        float32x4_t c10 = vld1q_f32(tileC_packed + 16);
        float32x4_t c11 = vld1q_f32(tileC_packed + 20);
        float32x4_t c12 = vld1q_f32(tileC_packed + 24);
        float32x4_t c13 = vld1q_f32(tileC_packed + 28);
        float32x4_t c20 = vld1q_f32(tileC_packed + 32);
        float32x4_t c21 = vld1q_f32(tileC_packed + 36);
        float32x4_t c22 = vld1q_f32(tileC_packed + 40);
        float32x4_t c23 = vld1q_f32(tileC_packed + 44);
        float32x4_t c30 = vld1q_f32(tileC_packed + 48);
        float32x4_t c31 = vld1q_f32(tileC_packed + 52);
        float32x4_t c32 = vld1q_f32(tileC_packed + 56);
        float32x4_t c33 = vld1q_f32(tileC_packed + 60);

        float32x4_t a = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f);
        float32x4_t b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f);
        float32x4_t b3 = vdupq_n_f32(0.0f);

        for (size_t k = 0; k < kc; k++) {
            a = vld1q_f32(tileA_packed);

            b0 = vld1q_f32(tileB_packed);
            b1 = vld1q_f32(tileB_packed + 4);
            b2 = vld1q_f32(tileB_packed + 8);
            b3 = vld1q_f32(tileB_packed + 12);

            c00 = vfmaq_laneq_f32(c00, b0, a, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a, 0);
            c03 = vfmaq_laneq_f32(c03, b3, a, 0);
            
            c10 = vfmaq_laneq_f32(c10, b0, a, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a, 1);
            c13 = vfmaq_laneq_f32(c13, b3, a, 1);

            c20 = vfmaq_laneq_f32(c20, b0, a, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a, 2);
            c23 = vfmaq_laneq_f32(c23, b3, a, 2);

            c30 = vfmaq_laneq_f32(c30, b0, a, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a, 3);
            c33 = vfmaq_laneq_f32(c33, b3, a, 3);

            tileA_packed += 4;
            tileB_packed += 16;
        }

        vst1q_f32(tileC_packed      , c00);
        vst1q_f32(tileC_packed + 4  , c01);
        vst1q_f32(tileC_packed + 8  , c02);
        vst1q_f32(tileC_packed + 12 , c03);
        vst1q_f32(tileC_packed + 16 , c10);
        vst1q_f32(tileC_packed + 20 , c11);
        vst1q_f32(tileC_packed + 24 , c12);
        vst1q_f32(tileC_packed + 28 , c13);
        vst1q_f32(tileC_packed + 32 , c20);
        vst1q_f32(tileC_packed + 36 , c21);
        vst1q_f32(tileC_packed + 40 , c22);
        vst1q_f32(tileC_packed + 44 , c23);
        vst1q_f32(tileC_packed + 48 , c30);
        vst1q_f32(tileC_packed + 52 , c31);
        vst1q_f32(tileC_packed + 56 , c32);
        vst1q_f32(tileC_packed + 60 , c33);
    }

    void kernel_fma_4x16(float32_t* __restrict tileA_packed, float32_t* __restrict tileB_packed, float32_t* __restrict C, size_t stride_c, size_t kc) {
        float32x4_t c00 = vld1q_f32(C);
        float32x4_t c01 = vld1q_f32(C + 4);
        float32x4_t c02 = vld1q_f32(C + 8);
        float32x4_t c03 = vld1q_f32(C + 12);
        float32x4_t c10 = vld1q_f32(C + stride_c);
        float32x4_t c11 = vld1q_f32(C + stride_c + 4);
        float32x4_t c12 = vld1q_f32(C + stride_c + 8);
        float32x4_t c13 = vld1q_f32(C + stride_c + 12);
        float32x4_t c20 = vld1q_f32(C + 2 * stride_c);
        float32x4_t c21 = vld1q_f32(C + 2 * stride_c + 4);
        float32x4_t c22 = vld1q_f32(C + 2 * stride_c + 8);
        float32x4_t c23 = vld1q_f32(C + 2 * stride_c + 12);
        float32x4_t c30 = vld1q_f32(C + 3 * stride_c);
        float32x4_t c31 = vld1q_f32(C + 3 * stride_c + 4);
        float32x4_t c32 = vld1q_f32(C + 3 * stride_c + 8);
        float32x4_t c33 = vld1q_f32(C + 3 * stride_c + 12);

        float32x4_t a = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f);
        float32x4_t b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f);
        float32x4_t b3 = vdupq_n_f32(0.0f);

        for (size_t k = 0; k < kc; k++) {
            a = vld1q_f32(tileA_packed);

            b0 = vld1q_f32(tileB_packed);
            b1 = vld1q_f32(tileB_packed + 4);
            b2 = vld1q_f32(tileB_packed + 8);
            b3 = vld1q_f32(tileB_packed + 12);

            c00 = vfmaq_laneq_f32(c00, b0, a, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a, 0);
            c03 = vfmaq_laneq_f32(c03, b3, a, 0);
            
            c10 = vfmaq_laneq_f32(c10, b0, a, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a, 1);
            c13 = vfmaq_laneq_f32(c13, b3, a, 1);

            c20 = vfmaq_laneq_f32(c20, b0, a, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a, 2);
            c23 = vfmaq_laneq_f32(c23, b3, a, 2);

            c30 = vfmaq_laneq_f32(c30, b0, a, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a, 3);
            c33 = vfmaq_laneq_f32(c33, b3, a, 3);

            tileA_packed += 4;
            tileB_packed += 16;
        }

        vst1q_f32(C , c00);
        vst1q_f32(C + 4, c01);
        vst1q_f32(C + 8, c02);
        vst1q_f32(C + 12, c03);
        vst1q_f32(C + stride_c, c10);
        vst1q_f32(C + stride_c + 4, c11);
        vst1q_f32(C + stride_c + 8, c12);
        vst1q_f32(C + stride_c + 12, c13);
        vst1q_f32(C + 2 * stride_c, c20);
        vst1q_f32(C + 2 * stride_c + 4, c21);
        vst1q_f32(C + 2 * stride_c + 8, c22);
        vst1q_f32(C + 2 * stride_c + 12, c23);
        vst1q_f32(C + 3 * stride_c, c30);
        vst1q_f32(C + 3 * stride_c + 4, c31);
        vst1q_f32(C + 3 * stride_c + 8, c32);
        vst1q_f32(C + 3 * stride_c + 12, c33);
    }

    void gemm_single_thread(
        const float32_t* __restrict A,
        const float32_t* __restrict B,
        float32_t* __restrict C,
        size_t M, size_t N, size_t K
    ) {
        // Allocate once per-thread, which is the main thread in this case,
        // that means as long as the main thread lives, the buffers live.
        alignas(128) static thread_local float32_t blockA_packed[MC * KC];
        alignas(128) static thread_local float32_t blockB_packed[KC * NC];
        // Stack buffer for small array.
        alignas(128) float32_t tileC_packed[MR * NR];

        for (size_t jc = 0; jc < N; jc += NC) {
            size_t nc = std::min(N - jc, NC);

            for (size_t kp = 0; kp < K; kp += KC) {
                size_t kc = std::min(K - kp, KC);
                pack_blockB(B + kp * N + jc, blockB_packed, N, kc, nc);

                for (size_t ic = 0; ic < M; ic += MC) {
                    size_t mc = std::min(M - ic, MC);
                    pack_blockA(A + ic * K + kp, blockA_packed, K, mc, kc);

                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = std::min(mc - ir, MR);

                        for (size_t jr = 0; jr < nc; jr += NR) {
                            size_t nr = std::min(nc - jr, NR);

                            if (mr == MR && nr == NR) {
                                kernel_fma_4x16(
                                    blockA_packed + ir * kc,
                                    blockB_packed + jr * kc,
                                    C + (ic + ir) * N + jc + jr,
                                    N,
                                    kc
                                );
                            } else {
                                float32_t* tileC = C + (ic + ir) * N + jc + jr;
                                pack_tileC(tileC, tileC_packed, N, mr, nr);
                                kernel_fma_4x16_edge(
                                    blockA_packed + ir * kc,
                                    blockB_packed + jr * kc,
                                    tileC_packed,
                                    kc
                                );
                                unpack_tileC(tileC, tileC_packed, N, mr, nr);
                            }
                        }
                    }

                }
            }
        }
    }
}