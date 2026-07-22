#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <vector>

namespace matmul {
    template <typename T>
    void matmul_rMajor_vector(
    const std::vector< std::vector<T> >& A,
    const std::vector< std::vector<T> >& B,
    std::vector< std::vector<T> >& C,
    size_t p, size_t q, size_t r);

    template <typename T>
    void matmul_cMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t p, size_t q, size_t r);

    template <typename T>
    void matmul_rMajor_ptr(
        const T* __restrict A,
        const T* __restrict B,
        T* __restrict C,
        size_t p, size_t q, size_t r);

}
#endif