#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <vector>
#include <cstddef>

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

#endif