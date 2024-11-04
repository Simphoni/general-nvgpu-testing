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
                                      fp16 *gemmC_ptr, // fp16 *lnA, fp16 *lnB,
                                      int gemmM, int gemmN, int gemmK) {
  using gemmTileM = _128;
  using gemmTileN = _128;
  using gemmTileK = _32;
  using gemmPipe = _3;
  const int thridx = threadIdx.x + threadIdx.y * blockDim.x;
  auto ctaCoord = make_coord(blockIdx.x, blockIdx.y, _);
  auto gemmTiler = Shape<gemmTileM, gemmTileN, gemmTileK>{};

  auto tensorA = make_tensor(make_gmem_ptr(gemmA_ptr),
                             make_layout(make_shape(gemmK, gemmM)));
  auto tensorB = make_tensor(make_gmem_ptr(gemmB_ptr),
                             make_layout(make_shape(gemmK, gemmN)));
  auto tensorC = make_tensor(make_gmem_ptr(gemmC_ptr),
                             make_layout(make_shape(gemmN, gemmM)));
  using GemmCopyIntrin4B = SM80_CP_ASYNC_CACHEGLOBAL<uint32_t>;
  // accumulate with FP32
  // using MMAIntrin = SM80_16x8x16_F32F16F16F32_TN;

  auto copyA = make_tiled_copy(Copy_Atom<Copy_Traits<GemmCopyIntrin4B>, fp16>{},
                               Layout<Shape<_16, Int<2 * kWarpNum>>>{},
                               Layout<Shape<_2, _2>>{});
  auto copyB = make_tiled_copy(Copy_Atom<Copy_Traits<GemmCopyIntrin4B>, fp16>{},
                               Layout<Shape<_16, Int<2 * kWarpNum>>>{},
                               Layout<Shape<_2, _2>>{});

  // tX for tensorX
  Tensor tAGmem = local_tile(tensorA, gemmTiler, ctaCoord, Step<_1, X, _1>{});
  // when masked, ctaCoord become (blockIdx.x, _), selecting all the column
  Tensor tBGmem = local_tile(tensorB, gemmTiler, ctaCoord, Step<X, _1, _1>{});
  Tensor tCGmem = local_tile(tensorC, gemmTiler, ctaCoord, Step<_1, _1, X>{});

  auto layoutASmem = make_layout(Shape<gemmTileK, gemmTileM, gemmPipe>{});
  auto layoutBSmem = make_layout(Shape<gemmTileK, gemmTileN, gemmPipe>{});
  __shared__ fp16 sA[cosize_v<decltype(layoutASmem)>];
  __shared__ fp16 sB[cosize_v<decltype(layoutBSmem)>];
  Tensor tASmem = make_tensor(make_smem_ptr(sA), layoutASmem);
  Tensor tBSmem = make_tensor(make_smem_ptr(sB), layoutBSmem);

  ThrCopy thrCopyA = copyA.get_slice(thridx);
  ThrCopy thrCopyB = copyB.get_slice(thridx);

  // cA for [copy]TiledTensorA
  Tensor cAGmem =
      thrCopyA.partition_S(tAGmem); // (COPY,COPY_M,COPY_K,k_tile_count)
  Tensor cASmem = thrCopyA.partition_D(tASmem);
  Tensor cBGmem = thrCopyB.partition_S(tBGmem);
  Tensor cBSmem = thrCopyB.partition_D(tBSmem);

  auto blockMMA = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                 Layout<Shape<_16, _8, _1>>{});
  ThrMMA thrMMA = blockMMA.get_slice(thridx);
  // mA for [mma]TiledTensorA
  Tensor mASmem = thrMMA.partition_A(tASmem); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor mBSmem = thrMMA.partition_B(tBSmem);
  Tensor mCGmem = thrMMA.partition_C(tCGmem);

  Tensor mAReg = thrMMA.make_fragment_A(mASmem(_, _, _, 0));
  Tensor mBReg = thrMMA.make_fragment_B(mBSmem(_, _, _, 0));
  Tensor mCReg = thrMMA.make_fragment_C(mCGmem);

  clear(mCReg);

  constexpr int kMaxPipe = size<3>(mASmem);
  int kTileCount = size<3>(cAGmem);
  int kMaxBlock = size<2>(mASmem);
  int loadIdx = 0;
  for (int pipeIdx = 0; pipeIdx < kTileCount + kMaxPipe - 1; pipeIdx++) {
    if (0 < pipeIdx && pipeIdx < kTileCount) {
      copy(copyA, cAGmem(_, _, _, pipeIdx), cASmem(_, _, _, loadIdx));
      copy(copyB, cBGmem(_, _, _, pipeIdx), cBSmem(_, _, _, loadIdx));
    }
    if (kMaxPipe - 1 <= pipeIdx) {
      int mmaIdx = loadIdx + 1;
      mmaIdx = mmaIdx == kMaxPipe ? 0 : mmaIdx;
      cp_async_wait<kMaxPipe - 2>();
      __syncthreads();
      for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
        copy(mASmem(_, _, blockIdx, mmaIdx), mAReg(_, _, blockIdx));
        copy(mBSmem(_, _, blockIdx, mmaIdx), mBReg(_, _, blockIdx));
        gemm(blockMMA, mAReg(_, _, mmaIdx), mBReg(_, _, mmaIdx), mCReg);
      }
    }
    loadIdx += 1;
    loadIdx = loadIdx == kMaxPipe ? 0 : loadIdx;
  }
  take<0, 1>(shape(mCReg));
  copy(mCReg, mCGmem);
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
  int blockX = gemmM / 128;
  int blockY = gemmN / 128;
  dim3 gridDim(blockX, blockY);
  dim3 blockDim(32, 4);
  fp16 *gemmA_ptr = gemmA.data_ptr<fp16>();
  fp16 *gemmB_ptr = gemmB.data_ptr<fp16>();
  fp16 *gemmC_ptr = gemmC.data_ptr<fp16>();
  cute_parallel_gemm_ln<<<gridDim, blockDim>>>(gemmA_ptr, gemmB_ptr, gemmC_ptr,
                                               gemmM, gemmN, gemmK);
}

void register_cutlass_parallel(pybind11::module &m) {
  m.def("cutlass_parallel_gemmrc_layernorm",
        &_cutlass_parallel_gemmrc_layernorm);
}
