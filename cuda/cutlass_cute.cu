#include "cute/tensor.hpp"
#include "pybind11/pybind11.h"

#include "gemm_utils.h"
#include "torch_utils.h"

using namespace cute;
using namespace cutlass;

namespace {

using fp16 = cutlass::half_t;

constexpr int kWarpNum = 4;

__global__ void cute_parallel_gemm_ln(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                      fp16 *gemmC_ptr, fp16 *lnA, fp16 *lnB,
                                      int gemmM, int gemmN, int gemmK) {
  auto probTiler = make_shape(_128{}, _128{}, _32{});
  auto ctaCoord = make_coord(blockIdx.x, blockIdx.y, _);

  auto tensorA = make_tensor(
      gemmA_ptr, make_layout(make_shape(gemmM, gemmK), LayoutRight{}));
  auto tensorB = make_tensor(gemmB_ptr, make_layout(make_shape(gemmK, gemmN)));
  auto tensorC = make_tensor(
      gemmC_ptr, make_layout(make_shape(gemmM, gemmN), LayoutRight{}));
  using GemmCopyIntrin4B = SM80_CP_ASYNC_CACHEGLOBAL<uint32_t>;
  // accumulate with FP32
  // using MMAIntrin = SM80_16x8x16_F32F16F16F32_TN;

  auto GemmACopy = make_tiled_copy(
      Copy_Atom<Copy_Traits<GemmCopyIntrin4B>, fp16>{},
      make_layout(make_shape(Int<2 * kWarpNum>{}, _16{}), LayoutRight{}),
      make_layout(make_shape(_8{}, _2{}), LayoutRight{}));

  auto GemmASmemLayout = make_layout(make_shape(_128{}, _32{}), LayoutRight{});
  __shared__ fp16 smemGemmA[cosize_v<decltype(GemmASmemLayout)>];

  ThrCopy thr_copy_a = GemmACopy.get_slice(threadIdx.x);
}

} // namespace

// parallel operator
// output:
//  - gemmC <- gemmA * gemmB
//  - lnB <- layernorm(lnA)
// layout:
// - 'c' for colmajor, 'r' for rowmajor
// - gemmC is row major, it could be important for the following operators
void _cutlass_parallel_gemmrc_layernorm(torch::Tensor gemmA,
                                        torch::Tensor gemmB, torch::Tensor gemmC
                                        // torch::Tensor ln_A,
                                        // torch::Tensor ln_B
) {
  // for gemmrc, A[m,k], B[n,k], C[m,n], (LayoutRight)
  int gemmM = gemmC.size(0);
  int gemmN = gemmC.size(1);
  int gemmK = gemmA.size(1);
}

void register_cutlass_parallel(pybind11::module &m) {
  m.def("cutlass_parallel_gemmrc_layernorm",
        &_cutlass_parallel_gemmrc_layernorm);
}
