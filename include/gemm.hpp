#ifndef GEMM_HPP
#define GEMM_HPP

#include <vector>
#include <cstddef>
#include <arm_neon.h>

class Gemm {
public:
    Gemm(size_t mc, size_t nc, size_t kc) : MC(mc), NC(nc), KC(kc) {
        blockA_packed.resize(MC * KC);
        blockB_packed.resize(KC * NC);
        tileC_packed.resize(MR * NR);
    }

    void operator()(
        const float32_t* __restrict A,
        const float32_t* __restrict B,
        float32_t* __restrict C,
        size_t M,
        size_t N,
        size_t K
    );

private:
    std::vector<float32_t> blockA_packed;
    std::vector<float32_t> blockB_packed;
    std::vector<float32_t> tileC_packed;
    size_t MC;
    size_t NC;
    size_t KC;
    size_t MR = 4;
    size_t NR = 16;

};

#endif