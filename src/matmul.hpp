#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <vector>
#include <arm_neon.h>

namespace matmul {
    template <typename T>
    void matmul_rMajor_vector(
    const std::vector< std::vector<T> >& A,
    const std::vector< std::vector<T> >& B,
    std::vector< std::vector<T> >& C,
    size_t M, size_t K, size_t N);

    template <typename T>
    void matmul_cMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t M, size_t K, size_t N);

    template <typename T>
    void matmul_rMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t M, size_t K, size_t N);

}

namespace gemm_f32_kernel4x16 {
    constexpr size_t MC = 16;
    constexpr size_t NC = 1024;
    constexpr size_t KC = 384;
    constexpr size_t MR = 4;
    constexpr size_t NR = 16;
    void pack_tileA(const float32_t* __restrict A, float32_t* __restrict tileA, size_t stride_a, size_t mr, size_t kc);
    void pack_blockA(const float32_t* __restrict A, float32_t* __restrict blockA, size_t stride_a, size_t mc, size_t kc);
    void pack_tileB(const float32_t* __restrict B, float32_t* __restrict tileB, size_t stride_b, size_t kc, size_t nr);
    void pack_blockB(const float32_t* __restrict B, float32_t* __restrict blockB, size_t stride_b, size_t kc, size_t nc);
    void pack_tileC(float32_t* __restrict C, float32_t* __restrict tileC, size_t stride_c, size_t mr, size_t nr);
    void unpack_tileC(float32_t* __restrict C, float32_t* __restrict tileC, size_t stride_c, size_t mr, size_t nr);
    void kernel_fma_4x16_edge(float32_t* __restrict tileA_packed, float32_t* __restrict tileB_packed, float32_t* __restrict tileC_packed, size_t kc);
    void kernel_fma_4x16(float32_t* __restrict tileA_packed, float32_t* __restrict tileB_packed, float32_t* __restrict C, size_t stride_c, size_t kc);
    void gemm_single_thread(const float32_t* __restrict A, const float32_t* __restrict B, float32_t* __restrict C, size_t M, size_t N, size_t K);

}
#endif