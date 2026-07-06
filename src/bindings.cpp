#include <pybind11/pybind11.h>
#include "env_test.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matmul_cpp, m) {
    m.doc() = "Python bindings for the matmul-perf-bridge C++ module";
    m.def("greet", &env_test::greet, "A simple greeting function.", py::arg("name") = "World");
}