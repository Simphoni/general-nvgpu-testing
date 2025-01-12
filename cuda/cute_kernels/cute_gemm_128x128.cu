#include "cute/tensor.hpp"

#include "cuda_utils.h"
#include "gemm_utils.h"

namespace parallel_kernels {

using namespace cute;
using namespace cutlass;
using fp16 = cute::half_t;

extern __shared__ uint8_t shmem_ptr[];

static __global__ void __launch_bounds__(128)
    kernel_cute_parallel_gemm(fp16 *gemmA_ptr, fp16 *gemmB_ptr, fp16 *gemmC_ptr,
                              int gemmM, int gemmN, int gemmK) {
  using gemmTileM = _128;
  using gemmTileN = _128;
  using gemmTileK = _32;
  using gemmPipe = _4;
  const int thridx = threadIdx.x + threadIdx.y * blockDim.x;
  int ctaIdx = blockIdx.x + blockIdx.y * gridDim.x;
  // mtile for 'Microtile'.
  // It's used for mapping between CTA and the Gemm sub-matrix,
  // targeting better L2 cache utilization.
  constexpr int mtileM = 8;
  int mtileSize = mtileM * gridDim.y;
  int mtileIdx = ctaIdx / mtileSize;
  int tileIdx = ctaIdx % mtileSize;
  int mtileMSize = min(mtileM, gridDim.x - mtileIdx * mtileM);
  int mIdx = mtileIdx * mtileM + tileIdx % mtileMSize;
  int nIdx = tileIdx / mtileMSize;
  auto ctaCoord = make_coord(mIdx, nIdx, _);
  // auto ctaCoord = make_coord(blockIdx.x, blockIdx.y, _);
  auto gemmTiler = Shape<gemmTileM, gemmTileN, gemmTileK>{};

  auto tensorA =
      make_tensor(make_gmem_ptr(gemmA_ptr),
                  make_layout(make_shape(gemmM, gemmK), LayoutRight{}));
  auto tensorB =
      make_tensor(make_gmem_ptr(gemmB_ptr),
                  make_layout(make_shape(gemmM, gemmK), LayoutRight{}));
  auto tensorC =
      make_tensor(make_gmem_ptr(gemmC_ptr),
                  make_layout(make_shape(gemmM, gemmN), LayoutRight{}));
  // CP_ASYNC only accept 16B, cutlass 3.5.1 assertion too loose
  using AsyncCopy =
      Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, fp16>;

  // when masked, ctaCoord become (blockIdx.x, _), selecting all the column
  Tensor tAGmem = local_tile(tensorA, gemmTiler, ctaCoord, Step<_1, X, _1>{});
  Tensor tBGmem = local_tile(tensorB, gemmTiler, ctaCoord, Step<X, _1, _1>{});
  Tensor tCGmem = local_tile(tensorC, gemmTiler, ctaCoord, Step<_1, _1, X>{});

  auto layoutASmem = composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<gemmTileM, gemmTileK, gemmPipe>{},
                  make_stride(gemmTileK{}, _1{}, gemmTileM{} * gemmTileK{})));
  auto layoutBSmem = composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<gemmTileN, gemmTileK, gemmPipe>{},
                  make_stride(gemmTileK{}, _1{}, gemmTileN{} * gemmTileK{})));
  auto layoutCSmem =
      composition(Swizzle<3, 4, 3>{},
                  make_layout(Shape<gemmTileM, gemmTileN>{}, LayoutRight{}));
  // [cosize_v<decltype(layoutASmem)>];
  fp16 *sA = (fp16 *)shmem_ptr;
  // [cosize_v<decltype(layoutBSmem)>];
  fp16 *sB = sA + cosize_v<decltype(layoutASmem)>;
  fp16 *sC = (fp16 *)shmem_ptr; // sA is out of scope when sC is alive
  Tensor tASmem = make_tensor(make_smem_ptr(sA), layoutASmem);
  Tensor tBSmem = make_tensor(make_smem_ptr(sB), layoutBSmem);
  Tensor tCSmem = make_tensor(make_smem_ptr(sC), layoutCSmem);

  auto copyA =
      make_tiled_copy(AsyncCopy{}, make_layout(Shape<_32, _4>{}, LayoutRight{}),
                      make_layout(Shape<_1, _8>{}, LayoutRight{}));
  auto copyB = copyA;

  // (COPY,COPY_M,COPY_K,k_tile_count)
  ThrCopy thrCopyA = copyA.get_slice(thridx);
  Tensor copyASrc = thrCopyA.partition_S(tAGmem);
  Tensor copyADst = thrCopyA.partition_D(tASmem);
  ThrCopy thrCopyB = copyB.get_slice(thridx);
  Tensor copyBSrc = thrCopyB.partition_S(tBGmem);
  Tensor copyBDst = thrCopyB.partition_D(tBSmem);

  // thr_layout tiles the work onto all warps
  auto blockMMA = make_tiled_mma(
      MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>{},
      make_layout(Shape<_2, _2>{}, LayoutRight{}),
      Tile<decltype(get<0>(gemmTiler)), decltype(get<1>(gemmTiler)), _16>{});
  ThrMMA thrMMA = blockMMA.get_slice(thridx);
  // partition_A expect (M, K, ...)
  Tensor mmaASmem = thrMMA.partition_A(tASmem);
  // partition_B expect (N, K, ...)
  Tensor mmaBSmem = thrMMA.partition_B(tBSmem);
  // partition_C expect (M, N, ...)
  Tensor mmaCSmem = thrMMA.partition_C(tCSmem);

  Tensor mmaAReg = thrMMA.make_fragment_A(mmaASmem(_, _, _, 0));
  Tensor mmaBReg = thrMMA.make_fragment_B(mmaBSmem(_, _, _, 0));
  Tensor mmaCReg = thrMMA.make_fragment_C(mmaCSmem);

  clear(mmaCReg);
  // this is the ldmatrix 8x8 command (on 4 matrices)
  using LdMatrix = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, fp16>;

  auto ldmA = make_tiled_copy_A(LdMatrix{}, blockMMA);
  auto thrLdmA = ldmA.get_slice(thridx);
  auto ldmASrc = thrLdmA.partition_S(tASmem);
  auto ldmADst = thrLdmA.retile_D(mmaAReg);
  auto ldmB = make_tiled_copy_B(LdMatrix{}, blockMMA);
  auto ldmThrB = ldmB.get_slice(thridx);
  auto ldmBSrc = ldmThrB.partition_S(tBSmem);
  auto ldmBDst = ldmThrB.retile_D(mmaBReg);

  constexpr int kMaxPipe = size<3>(mmaASmem);
  int kTileCount = size<3>(copyASrc);
  int kMaxBlock = size<2>(mmaASmem);
  int loadIdx = 0;

  {
    int pipeIdx = 0;
    int mmaIdx = 0;
    copy(copyA, copyASrc(_, _, _, pipeIdx), copyADst(_, _, _, loadIdx));
    copy(copyB, copyBSrc(_, _, _, pipeIdx), copyBDst(_, _, _, loadIdx));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    CUTE_UNROLL
    for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
      copy(ldmA, ldmASrc(_, _, blockIdx, mmaIdx), ldmADst(_, _, blockIdx));
      copy(ldmB, ldmBSrc(_, _, blockIdx, mmaIdx), ldmBDst(_, _, blockIdx));
      gemm(blockMMA, mmaAReg(_, _, blockIdx), mmaBReg(_, _, blockIdx), mmaCReg);
    }
    loadIdx = 1;
  }

  for (int pipeIdx = 1; pipeIdx < kTileCount + kMaxPipe - 1; pipeIdx++) {
    if (pipeIdx < kTileCount) {
      copy(copyA, copyASrc(_, _, _, pipeIdx), copyADst(_, _, _, loadIdx));
      copy(copyB, copyBSrc(_, _, _, pipeIdx), copyBDst(_, _, _, loadIdx));
      cp_async_fence();
    }
    if (kMaxPipe <= pipeIdx) {
      int mmaIdx = loadIdx + 1;
      mmaIdx = mmaIdx == kMaxPipe ? 0 : mmaIdx;
      cp_async_wait<kMaxPipe - 1>();
      __syncthreads(); // loading from shared requires barrier
      CUTE_UNROLL
      for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
        copy(ldmA, ldmASrc(_, _, blockIdx, mmaIdx), ldmADst(_, _, blockIdx));
        copy(ldmB, ldmBSrc(_, _, blockIdx, mmaIdx), ldmBDst(_, _, blockIdx));
        gemm(blockMMA, mmaAReg(_, _, blockIdx), mmaBReg(_, _, blockIdx),
             mmaCReg);
      }
    }
    loadIdx += 1;
    loadIdx = loadIdx == kMaxPipe ? 0 : loadIdx;
  }

  Tensor mmaCRegFp16 = make_fragment_like<fp16>(mmaCReg.layout());
  using StMatrix = Copy_Atom<UniversalCopy<AlignedArray<fp16, 2>>, fp16>;
  using Copy8B = Copy_Atom<UniversalCopy<AlignedArray<fp16, 4>>, fp16>;
  auto stmC = make_tiled_copy_C(StMatrix{}, blockMMA);
  // auto thrStmC = stmC.get_slice(thridx);
  // auto stmCSrc = thrStmC.retile_S(mmaCRegFp16);
  // auto stmCDst = thrStmC.partition_D(tCSmem);
  auto copyC =
      make_tiled_copy(Copy8B{}, make_layout(Shape<_4, _32>{}, LayoutRight{}),
                      make_layout(Shape<_1, _4>{}, LayoutRight{}));
  auto thrCopyC = copyC.get_slice(thridx);
  auto copyCSrc = thrCopyC.partition_S(tCSmem);
  auto copyCDst = thrCopyC.partition_D(tCGmem);
  for (int j = 0; j < size<2>(mmaCReg); j++) {
    auto regFp16 = mmaCRegFp16(_, _, j);
    auto regFp32 = mmaCReg(_, _, j);
    auto smem = mmaCSmem(_, _, j);
    __half2 packedHalf;
    CUTE_UNROLL
    for (int i = 0; i < size(regFp32); i += 2) {
      packedHalf = __floats2half2_rn(regFp32[i], regFp32[i + 1]);
      regFp16[i] = packedHalf.x;
      regFp16[i + 1] = packedHalf.y;
    }
    copy(stmC, regFp16, smem);
  }
  __syncthreads();
  copy(copyC, copyCSrc, copyCDst);
}

void entry_custom_gemmrc_128x128(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                 fp16 *gemmC_ptr, int gemmM, int gemmN,
                                 int gemmK) {
  dim3 gridDim(gemmM / 128, gemmN / 128);
  dim3 blockDim(128, 1, 1);

  constexpr int SMEM_SIZE = 64 * 1024;
  cudaSafeCall(cudaFuncSetAttribute(parallel_kernels::kernel_cute_parallel_gemm,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    SMEM_SIZE));
  parallel_kernels::kernel_cute_parallel_gemm<<<gridDim, blockDim, SMEM_SIZE>>>(
      gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK);
}

} // namespace parallel_kernels
