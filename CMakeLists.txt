cmake_minimum_required(VERSION 3.14)
project(matmul-perf-bridge)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 2.13 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, Fetching pybind11 from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY  https://github.com/pybind/pybind11
        GIT_TAG         v2.13.6
        GIT_SHALLOW     TRUE
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "pybind11 found: ${pybind11_DIR}")
endif()

pybind11_add_module(matmul_cpp 
    src/bindings.cpp
    src/matmul.cpp
    src/gemm.cpp
)

target_include_directories(matmul_cpp
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(matmul_cpp
    PRIVATE
        -O3
)
