# matmul-perf-bridge

This is a learning project exploring how to package a C++ extension module for Python using `pybind11` and `CMake`, with matrix multiplication as the target workload.

After bridging C++ and Python, the interesting part lies in how to optimize the performance, and that leads to BLAS (Basic Linear Algebra Subprogram). Rather than digging into the primary BLAS literature, I found and followed a very helpful step-by-step tutorial by [solykova](https://salykova.github.io/gemm-cpu), which covers theoretical and practical implementation on C.

Instead of running on Ubuntu with an AMD Ryzen like solykova, my environment is macOS on M1. Therefore, `arm:neon` is applied in my GEMM (General Matrix Multiplication) kernel.

## Status and Findings

Custom GEMM complete, together with three C++ approaches (nested `std::vector`, row/column-major flat pointer), benchmarked against NumPy's `matmul` and a Python triple-loop across matrix sizes from 100x100x100 to 6400x6400x6400.

- [x] Get a C++ module building and importable from Python
- [x] Implement matrix multiplication in C++, and pass data between C++ and Python
- [x] Benchmark against NumPy (to quantify the C++ speedup, if any)
- [x] Explore optimization (SIMD, blocking, threading, etc.)

Here are some key things I found during benchmarking and tuning:

- **Warm-up matters**: In an earlier version, I timed each algorithm once per matrix size with no warm-up. NumPy's *first* call was slower than its later calls, even on 8x more FLOPs — a sign the first call was absorbing one-time costs (BLAS thread-pool initialization, dynamic symbol resolution, CPU frequency ramp-up). Since these costs are process-level rather than per-matrix-size, a single warm-up pass before the real benchmarking begins is enough to absorb them.

- **Round-robin/shuffled sweep**: While tuning panel sizes on the multi-threaded GEMM in [gemm_multiThreads](https://github.com/ditatteda01/gemm_multiThreads), I ran into a first-runner-wins pattern — the first (MC, NC, KC) combination tested consistently outperformed later ones, with a clear slowdown trend as testing went on. This traced back to P-core frequency throttling under sustained thermal load. I adopted the same fix here: instead of running each algorithm 5 times back-to-back before moving to the next, each epoch now sweeps all algorithms once per round, repeated 5 times, and takes the median per algorithm afterward. This dilutes any thermal drift across algorithms rather than concentrating it within one.

*Note: This project focused on the Python/C++ pipeline and a single-threaded GEMM under light load. Multi-threaded GEMM implementation, block-size tuning, and the thermal-throttling investigation live in the separate [gemm_multiThreads](https://github.com/ditatteda01/gemm_multiThreads) project, kept apart specifically because that thermal confound made heavy-load tuning a different methodological problem than this one.*

## Structure

```text
matmul-perf-bridge/
│
├── doc/
│   └── benchmark_results.md
│
├── log/
│   └── metrics.log             # Raw benchmark output
│
├── png/
│   ├── benchmark_cpp_GFLOPs.png
│   ├── benchmark_total_time.png
│   └── benchmark_cpp_alg_time.png
│
├── scripts/
│   ├── env_test.py             # Execute test for C++/Python bridge
│   ├── benchmark.py            # Execute benchmarks
│   ├── parser.py               # Parser of log
│   └── plot_metrics.py         # Generate benchmark figures
│
├── include/
│   ├── env_test.hpp            # Testing C++/Python bridge
│   ├── gemm_kernel.hpp         # GEMM tool
│   ├── gemm_pack.hpp           # GEMM tool
│   ├── gemm.hpp                # GEMM interface
│   └── matmul.hpp              # Nested vector, row/column-major
│
├── src/
│   ├── bindings.cpp            # pybind11 wrappers exposing methods to Python
│   ├── matmul.cpp
│   └── gemm.cpp
│
├── CMakeLists.txt
│
└── README.md
```

## Requirements

- Compiler: Apple Clang 17.0.0
- Python: 3.11
- CMake: 3.14 or later

## Build

In this project, I hand off all the work to CMake through `FetchContent`, so CMake takes care of the module under `build/_deps/` after configuration, which makes the project much more portable. (*Please refer to [example_pybind11](https://github.com/ditatteda01/example_pybind11) for more details on how to download the module and set dependencies manually.*)

Here's the `CMakeLists.txt` snippet:

```CMake
find_package(pybind11 2.13 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY  https://github.com/pybind/pybind11
        GIT_TAG         v2.13.6
        GIT_SHALLOW     TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()
```

1. **Configure CMake and Build**

Make sure your Python virtual environment is activated, so CMake picks up the
correct Python interpreter. In the terminal, navigate to the project root and run:

```bash
source .venv/bin/activate
rm -rf build   # clear any stale CMake cache from a previous Python version
cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
cmake --build build
python ./scripts/env_test.py
```

*Note: If you see `ModuleNotFoundError` after building, it usually means CMake configured against a different Python (probably the global one) than the one you're running — re-check with `which python` and rebuild.*

2. **Test C++/Python Bridge**

Run the pipeline test between Python and C++:

```bash
python3 ./scripts/env_test.py
```

You should see greetings in your terminal.

## Running the benchmark

Runs matrix multiplication across a range of increasing sizes (doubling all three dimensions each epoch) with methods: NumPy's `matmul`, Python triple-loop, three naive C++ loop implementations, and one C++ custom GEMM.

```bash
python ./scripts/benchmark.py           # produce metrics.log
python ./scripts/plot_metrics.py        # regenerate figures based on metrics.log
```

See [`doc/benchmark_results.md`](doc/benchmark_results.md) for implementation details, key findings, and benchmark methodology notes.

## License

For learning purposes, no license.
