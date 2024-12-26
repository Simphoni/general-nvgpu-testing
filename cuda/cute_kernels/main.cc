#include <cassert>

#include "cute/tensor.hpp"

#include "cuda_utils.h"
#include "gemm_utils.h"
#include "torch_utils.h"

using fp16 = cute::half_t;

namespace parallel_kernels {

void entry_cute_parallel_gemmrc(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                     fp16 *gemmC_ptr, int gemmM, int gemmN,
                                     int gemmK);
void entry_cute_parallel_gemmrc_lnr(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                    fp16 *gemmC_ptr, int gemmM, int gemmN,
                                    int gemmK, fp16 *lnA_ptr, fp16 *lnB_ptr,
                                    int lnM, int lnN);

} // namespace parallel_kernels

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

void _cutlass_parallel_gemmrc(torch::Tensor gemmA, torch::Tensor gemmB,
                                   torch::Tensor gemmC) {
  // for gemmrc, A[m,k], B[n,k], C[m,n], (LayoutRight)
  int gemmM = gemmC.size(0);
  int gemmN = gemmC.size(1);
  int gemmK = gemmA.size(1);
  fp16 *gemmA_ptr = reinterpret_cast<fp16 *>(gemmA.data_ptr());
  fp16 *gemmB_ptr = reinterpret_cast<fp16 *>(gemmB.data_ptr());
  fp16 *gemmC_ptr = reinterpret_cast<fp16 *>(gemmC.data_ptr());
  parallel_kernels::entry_cute_parallel_gemmrc(
      gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK);
}

void _cutlass_parallel_gemmrc_lnr(torch::Tensor gemmA, torch::Tensor gemmB,
                                  torch::Tensor gemmC, torch::Tensor lnA,
                                  torch::Tensor lnB) {
  int gemmM = gemmC.size(0);
  int gemmN = gemmC.size(1);
  int gemmK = gemmA.size(1);
  assert(gemmA.size(0) == gemmM);
  assert(gemmB.size(0) == gemmN);
  assert(gemmB.size(1) == gemmK);
  fp16 *gemmA_ptr = reinterpret_cast<fp16 *>(gemmA.data_ptr());
  fp16 *gemmB_ptr = reinterpret_cast<fp16 *>(gemmB.data_ptr());
  fp16 *gemmC_ptr = reinterpret_cast<fp16 *>(gemmC.data_ptr());
  int lnM = lnA.size(0);
  int lnN = lnA.size(1);
  assert(lnM == lnB.size(0));
  assert(lnN == lnB.size(1));
  fp16 *lnA_ptr = reinterpret_cast<fp16 *>(lnA.data_ptr());
  fp16 *lnB_ptr = reinterpret_cast<fp16 *>(lnB.data_ptr());
  parallel_kernels::entry_cute_parallel_gemmrc_lnr(
      gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK, lnA_ptr, lnB_ptr,
      lnM, lnN);
}

void register_cute_kernels(pybind11::module &mod) {
  mod.def("cutlass_parallel_gemmrc_lnr", &_cutlass_parallel_gemmrc_lnr);
  mod.def("cutlass_parallel_gemmrc", &_cutlass_parallel_gemmrc);
}
