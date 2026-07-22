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
        size_t p, size_t q, size_t r) {
            for (size_t i = 0; i < p; i++) {
                for (size_t k = 0; k < q; k++) {
                    for (size_t j = 0; j < r; j++) {
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
        size_t p, size_t q, size_t r) {
        for (size_t i = 0; i < p; i++) {
            for (size_t j = 0; j < r; j++) {
                for (size_t k = 0; k < q; k++) {
                    C[i*r + j] += A[i*q + k] * B[k*r + j];
                }
            }
        }
    }

    template <typename T>
    void matmul_rMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t p, size_t q, size_t r) {
        for (size_t i = 0; i < p; i++) {
            for (size_t k = 0; k < q; k++) {
                T valA = A[i*q + k];
                for (size_t j = 0; j < r; j++) {
                    C[i*r + j] += valA * B[k*r + j];
                }
            }
        }
    }

    // Force Compiler to generate compiled binary code so that dlopen can find the symbols at runtime.
    template void matmul_rMajor_vector(const std::vector< std::vector<int32_t> >& A, const std::vector< std::vector<int32_t> >& B, std::vector< std::vector<int32_t> >& C, size_t p, size_t q, size_t r);
    template void matmul_rMajor_vector(const std::vector< std::vector<double> >& A, const std::vector< std::vector<double> >& B, std::vector< std::vector<double> >& C, size_t p, size_t q, size_t r);
    template void matmul_cMajor_ptr<double>(const double*, const double*, double*, size_t, size_t, size_t);
    template void matmul_cMajor_ptr<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t);
    template void matmul_rMajor_ptr<double>(const double*, const double*, double*, size_t, size_t, size_t);
    template void matmul_rMajor_ptr<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t);

}
