#include <cstdio>

#include <pybind11/pybind11.h>

#include "cuda_utils.h"
#include "torch_utils.h"

void log_cublas_version();
void log_cutlass_version();

void _cublas_gemm_nt(at::Tensor a, at::Tensor b, at::Tensor c);

void _cublas_gemmex_nt_compf32(at::Tensor a, at::Tensor b, at::Tensor c);

void _cutlass_gemm_nt_naive(at::Tensor a, at::Tensor b, at::Tensor c);

// the exported functions should be named with a leading underscore
PYBIND11_MODULE(INTERFACE_NAME, m) {
  log_cublas_version();
  log_cutlass_version();
  m.def("cublas_gemm_nt", &_cublas_gemm_nt, "cublas_gemm_nt");
  m.def("cublas_gemmex_nt", &_cublas_gemmex_nt_compf32, "cublas_gemmex_nt");
  m.def("cutlass_gemm_nt_naive", &_cutlass_gemm_nt_naive,
        "cutlass_gemm_nt_naive");
}