#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>
#include "env_test.hpp"
#include "matmul.hpp"

namespace py = pybind11;

template <typename T>
std::pair<double, std::vector< std::vector<T> > > matmul_rMajor_vector_wrapper(
    const py::array_t<T> A,
    const py::array_t<T> B
) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional.");
    }
    if (A.shape(1) != B.shape(0)) {
        throw std::invalid_argument("Matrix inner dimensions dismatched. (A columns must equal B rows)");
    }

    // Create a copy of input array if it's memory discontiguous or type nonparity.
    auto valid_A = py::array_t<T, py::array::c_style>::ensure(A);
    auto valid_B = py::array_t<T, py::array::c_style>::ensure(B);
    if (!valid_A || !valid_B) {
        throw std::runtime_error("Failed to normalize input array buffers to contiguous block.");
    }

    size_t p = valid_A.shape(0);
    size_t q = valid_A.shape(1);
    size_t r = valid_B.shape(1);

    py::buffer_info bufA = valid_A.request();
    py::buffer_info bufB = valid_B.request();

    T* A_ptr = static_cast<T*>(bufA.ptr);
    T* B_ptr = static_cast<T*>(bufB.ptr);

    // Timer starts here because converting the flat NumPy buffer into
    // std::vector<std::vector<T>> (row-by-row heap allocation + copy) IS
    // part of the cost this method is meant to measure. Excluding it would
    // hide the very overhead that distinguishes this method from the
    // pointer-based ones.
    auto start = std::chrono::high_resolution_clock::now();

    std::vector< std::vector<T> > A_vec(p, std::vector<T>(q));
    for (size_t i = 0; i < p; i++) {
        std::copy(A_ptr + i * q, A_ptr + (i + 1) * q, A_vec[i].begin());
    }

    std::vector< std::vector<T> > B_vec(q, std::vector<T>(r));
    for (size_t k = 0; k < q; k++) {
        std::copy(B_ptr + k * r, B_ptr + (k + 1) * r, B_vec[k].begin());
    }

    std::vector< std::vector<T> > C(p, std::vector<T>(r));

    matmul::matmul_rMajor_vector<T>(A_vec, B_vec, C, p, q, r);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alg_elapsed = end - start;

    return {alg_elapsed.count(), C};
}

/* INFO: py::array_t constructor vs. C++ origin new

py::array_t constructs memory directly on the Python heap, leaving 'C' as 
a lightweight stack handle. When returned, Python handles garbage collection 
without manual memory management.

If using 'new' instead, pass a py::capsule base to prevent memory leaks by 
transferring ownership to Python:

```cpp
    T* C = new T[p * r]{};
    py::capsule gc_callback(
        C,
        [](void* ptr) {
            delete[] static_cast<T*>(ptr);
        }
    );
    return py::array_t<T>(shape, strides, C, gc_callback);
```
*/
template <typename T>
std::pair<double, py::array_t<T> > matmul_cMajor_ptr_wrapper(
    const py::array_t<T> A,
    const py::array_t<T> B
) {  
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional.");
    }
    if (A.shape(1) != B.shape(0)) {
        throw std::invalid_argument("Matrix inner dimensions dismatched. (A columns must equal B rows)");
    }

    // ensure() on an already-contiguous, correctly-typed array is a no-op
    // (just a refcount bump, no copy), so this step carries negligible cost
    // for this method and is deliberately left outside the timer. This
    // mirrors how the vector wrapper excludes its own no-cost setup (shape
    // validation) but includes its real conversion cost.
    auto valid_A = py::array_t<T, py::array::c_style>::ensure(A);
    auto valid_B = py::array_t<T, py::array::c_style>::ensure(B);
    if (!valid_A || !valid_B) {
        throw std::runtime_error("Failed to normalize input array buffers to contiguous block.");
    }

    size_t p = valid_A.shape(0);
    size_t q = valid_A.shape(1);
    size_t r = valid_B.shape(1);

    py::buffer_info bufA = valid_A.request();
    py::buffer_info bufB = valid_B.request();

    T* A_ptr = static_cast<T*>(bufA.ptr);
    T* B_ptr = static_cast<T*>(bufB.ptr);

    // Timer starts here because allocating C and zero-filling it IS part of
    // the cost this method is meant to measure — it's the pointer-method
    // equivalent of the vector wrapper's C(p, vector<T>(r)) allocation.
    // Keeping both methods' "build C" step inside their respective timers
    // ensures neither is unfairly credited for doing that work "for free."
    auto start = std::chrono::high_resolution_clock::now();

    py::array_t<T> C({p, r});
    py::buffer_info bufC = C.request();
    T* C_ptr = static_cast<T*>(bufC.ptr);
    std::fill(C_ptr, C_ptr + (p * r), T{0});

    matmul::matmul_cMajor_ptr<T>(A_ptr, B_ptr, C_ptr, p, q, r);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alg_elapsed = end - start;

    return {alg_elapsed.count(), C};
}

template <typename T>
std::pair<double, py::array_t<T> > matmul_rMajor_ptr_wrapper(
    const py::array_t<T> A,
    const py::array_t<T> B
) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional.");
    }
    if (A.shape(1) != B.shape(0)) {
        throw std::invalid_argument("Matrix inner dimensions dismatched. (A columns must equal B rows)");
    }

    // ensure() on an already-contiguous, correctly-typed array is a no-op
    // (just a refcount bump, no copy), so this step carries negligible cost
    // for this method and is deliberately left outside the timer. This
    // mirrors how the vector wrapper excludes its own no-cost setup (shape
    // validation) but includes its real conversion cost.
    auto valid_A = py::array_t<T, py::array::c_style>::ensure(A);
    auto valid_B = py::array_t<T, py::array::c_style>::ensure(B);
    if (!valid_A || !valid_B) {
        throw std::runtime_error("Failed to normalize input array buffers to contiguous block.");
    }

    size_t p = valid_A.shape(0);
    size_t q = valid_A.shape(1);
    size_t r = valid_B.shape(1);

    py::buffer_info bufA = valid_A.request();
    py::buffer_info bufB = valid_B.request();

    T* A_ptr = static_cast<T*>(bufA.ptr);
    T* B_ptr = static_cast<T*>(bufB.ptr);

    // Timer starts here because allocating C and zero-filling it IS part of
    // the cost this method is meant to measure — it's the pointer-method
    // equivalent of the vector wrapper's C(p, vector<T>(r)) allocation.
    // Keeping both methods' "build C" step inside their respective timers
    // ensures neither is unfairly credited for doing that work "for free."
    auto start = std::chrono::high_resolution_clock::now();

    py::array_t<T> C({p, r});
    py::buffer_info bufC = C.request();
    T* C_ptr = static_cast<T*>(bufC.ptr);
    std::fill(C_ptr, C_ptr + (p * r), T{0});

    matmul::matmul_rMajor_ptr<T>(A_ptr, B_ptr, C_ptr, p, q, r);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alg_elapsed = end - start;

    return {alg_elapsed.count(), C};
}

PYBIND11_MODULE(matmul_cpp, m) {
    m.doc() = "Python bindings for the matmul-perf-bridge C++ module";
    m.def("greet", &env_test::greet, "A simple greeting function.", py::arg("name") = "World");

    m.def("matmul_rMajor_vector", &matmul_rMajor_vector_wrapper<int32_t>, "Int32 matrix multiplication function.");
    m.def("matmul_rMajor_vector", &matmul_rMajor_vector_wrapper<double>, "Float64 matrix multiplication function.");

    m.def("matmul_cMajor_ptr", &matmul_cMajor_ptr_wrapper<int32_t>, "Int32 matrix multiplication function.");
    m.def("matmul_cMajor_ptr", &matmul_cMajor_ptr_wrapper<double>, "Float64 matrix multiplication function.");

    m.def("matmul_rMajor_ptr", &matmul_rMajor_ptr_wrapper<int32_t>, "Int32 matrix multiplication function.");
    m.def("matmul_rMajor_ptr", &matmul_rMajor_ptr_wrapper<double>, "Float64 matrix multiplication function.");

}