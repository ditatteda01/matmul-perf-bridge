#include "env_test.hpp"
#include "matmul.hpp"
#include <arm_neon.h>

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