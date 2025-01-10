# General NVIDIA GPU Testing

This repository holds **BEAM** (Basic Engine for Auto-tunable Matrix-multiplication).

## Implemented Tests

### Dense GEMM

This test covers the most common MatMul use case. The implementations listed below all follow the same algorithm, but with different optimizations.

Regarding the sources, there are roughly 3 types of implementations:

- **cuBLAS**: The cuBLAS library is the official NVIDIA library for linear algebra operations. It is highly optimized and uses the GPU to its full potential. cuBLAS is efficent with various matrix shapes and datatypes, due to the tremendous effort of hand crafted tunings, which makes it the most common choice for current deep learning frameworks.
    - `cublas_gemmexrc`
    - `cublas_hgemmrc`

- **cutlass**: The cutlass library is a collection of CUDA C++ templates and abstractions for implementing high-performance matrix-multiplication (GEMM) at all levels and scales within CUDA. The cutlass library can leveraged to implement high performance GEMM kernels, supporting a range of prologues, epilogues, and dequantization operations.
    - `cutlass_gemmrc`
    - `cutlass_gemmrc_spec`

- **cute**: The cute library is the underlying library for cutlass. It provides abstraction for both tensor layouts and cuda intrinsics. It is frequently used to craft custom kernels (e.g. FlashAttention) which uses Tensor Core Units.
    - `custom_gemmrc_128x128`
    - `custom_gemmrc_128x256`

### Dense GEMM with small M and N

The algorithm for this type of GEMM is called "split-k". It is a technique to split the k dimension into smaller chunks, which can be used to improve the performance of the GEMM operation by further breaking down the problem into smaller sub-problems to better utilize GPU SMs.

- **cutlass** implementations
    - `cutlass_gemmrc_splitk`
    - `cutlass_gemmrc_splitk_spec`