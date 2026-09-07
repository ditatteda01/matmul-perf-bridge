#include "gemm.hpp"
#include "gemm_kernel.hpp"
#include "gemm_pack.hpp"

void Gemm::operator()(
    const float32_t* __restrict A,
    const float32_t* __restrict B,
    float32_t* __restrict C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t jp = 0; jp < N; jp += NC) {
        size_t nc = std::min(N - jp, NC);

        for (size_t kp = 0; kp < K; kp += KC) {
            size_t kc = std::min(K - kp, KC);
            pack_blockB(
                B + kp * N + jp,
                blockB_packed.data(),
                N, kc, nc, NR
            );

            for (size_t ip = 0; ip < M; ip += MC) {
                size_t mc = std::min(M - ip, MC);
                pack_blockA(
                    A + ip * K + kp,
                    blockA_packed.data(),
                    K, mc, kc, MR
                );

                for (size_t ir = 0; ir < mc; ir += MR) {
                    size_t mr = std::min(mc - ir, MR);

                    for (size_t jr = 0; jr < nc; jr += NR) {
                        size_t nr = std::min(nc - jr, NR);

                        if (mr == MR && nr == NR) {
                            kernel_fma_4x16(
                                blockA_packed.data() + ir * kc,
                                blockB_packed.data() + kc * jr,
                                C + (ip + ir) * N + jp + jr,
                                N, kc
                            );

                        } else {
                            float* C_ = C + (ip + ir) * N + jp + jr;
                            pack_tileC(C_, tileC_packed.data(), N, mr, nr, MR, NR);
                            kernel_fma_4x16_edge(
                                blockA_packed.data() + ir * kc,
                                blockB_packed.data() + kc * jr,
                                tileC_packed.data(),
                                kc
                            );
                            unpack_tileC(C_, tileC_packed.data(), N, mr, nr, NR);
                        }
                    }
                }
            }
        }
    }
}