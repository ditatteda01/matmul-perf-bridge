# matmul-perf-bridge

  A learning project exploring how to package a C++ extension module for Python
using pybind11 and CMake, with matrix multiplication as the target workload.

## Status
  Work in progress. The build/binding pipeline (C++ → pybind11 → Python) is
working end-to-end with a placeholder `greet()` function. The matmul
implementation itself hasn't been added yet.

## Structure
- `src/` — C++ source (implementation, headers, pybind11 bindings)
- `scripts/` — Python-side test/entry scripts
- `CMakeLists.txt` — build configuration

## Goals
- [x] Get a C++ module building and importable from Python
- [ ] Implement matrix multiplication in C++, and pass data between C++ and Python
- [ ] Benchmark against NumPy (to quantify the C++ speedup, if any)
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