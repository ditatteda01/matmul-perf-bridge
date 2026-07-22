# matmul-perf-bridge

A learning project exploring how to package a C++ extension module for Python
using pybind11 and CMake, with matrix multiplication as the target workload.

## Status
Core functionality complete. Matrix multiplication is implemented in C++ using
three different approaches (nested `std::vector`, row-major flat pointer,
column-major flat pointer) and benchmarked against NumPy and a naive Python
loop across matrix sizes from 50×30×10 up to 3200×1920×640.

## Workflow

```text
Benchmark
    |
    ▼
benchmark.py
    |
    ▼
metrics.log
    ├────────────────► update_benchmark.py
    |                           |
    |                           ▼
    |                  benchmark_results.md
    |
    └────────────────► plot_metrics.py
                                |
                                ▼
                    benchmark_total_time.png
                    benchmark_cpp_alg_time.png
```

## Structure

```text
matmul-perf-bridge/
├── docs/
│   ├── metrics.log             # Raw benchmark output
│   ├── benchmark_results.md    # Benchmark results
│   ├── benchmark_total_time.png
│   └── benchmark_cpp_alg_time.png
│
├── scripts/
│   ├── benchmark.py            # Execute benchmarks
│   ├── update_benchmark.py     # Update benchmark_results.md
│   ├── plot_metrics.py         # Generate benchmark figures
│   ├── parser.py               # Shared log parser
│   └── env_test.py
|
├── src/
│   ├── bindings.cpp            # pybind11 wrappers exposing methods to Python
│   ├── matmul.cpp              # Core matrix multiplication algorithms
│   ├── matmul.hpp
│   └── env_test.hpp
│
├── CMakeLists.txt
│
└── README.md
```

*Note: `env_test.py` / `env_test.hpp` provide a minimal sanity check — a `greet()` function exposed through the full C++ → pybind11 → Python pipeline — used to confirm the build and binding setup work before trusting any matmul output.*

## Goals
- [x] Get a C++ module building and importable from Python
- [x] Implement matrix multiplication in C++, and pass data between C++ and Python
- [x] Benchmark against NumPy (to quantify the C++ speedup, if any)
- [ ] Explore optimization (SIMD, blocking, threading, etc.)

## Build

Make sure your virtual environment is activated, so CMake picks up the
correct Python interpreter.

```bash
source .venv/bin/activate
rm -rf build   # clear any stale CMake cache from a previous Python version
cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
cmake --build build
python ./scripts/env_test.py
```

*Note: run all steps from the project root. If you see `ModuleNotFoundError`
after building, it usually means CMake configured against a different Python
than the one you're running — re-check with `which python` and rebuild.*

## Running the benchmark

```bash
python ./scripts/benchmark.py           # produce metrics.log
python ./scripts/update_benchmark.py    # update benchmark_results.md based on metrics.log
python ./scripts/plot_metrics.py        # regenerate figures based on metrics.log
```

Runs matrix multiplication across a range of increasing sizes (doubling
roughly each epoch) using five methods: NumPy's `matmul`, a naive Python
triple-loop, and three C++ implementations. Each C++ method's result is
checked against a ground-truth matrix for correctness before timing is
reported.

See [`docs/benchmark_results.md`](docs/benchmark_results.md) for implementation details, key findings, and benchmark methodology notes.

## Next steps

Exploring cache blocking/tiling for the row-major pointer implementation, to
see how much of the remaining gap to NumPy/BLAS can be closed with a
hand-written tiled algorithm before reaching for SIMD intrinsics or threading.
