cmake_minimum_required(VERSION 3.20)
project(GeneralNVGPUTesting)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_CUDA_COMPILER_LAUNCHER "ccache")

add_subdirectory(cuda)
