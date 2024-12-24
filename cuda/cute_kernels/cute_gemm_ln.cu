#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"

#include "cuda_utils.h"
#include "gemm_utils.h"

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

namespace parallel_kernels {

using namespace cute;
using namespace cutlass;
using fp16 = cute::half_t;

extern __shared__ uint8_t shmem_ptr[];

#define ALIGN_UP(x, size) (((x) + (size)-1) / (size) * (size))
#define DIV_UP(x, size) (((x) + (size)-1) / (size))

static __global__ void __launch_bounds__(128)
    kernel_cute_parallel_gemm_ln(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                 fp16 *gemmC_ptr, int gemmM, int gemmN,
                                 int gemmK, fp16 *lnA_ptr, fp16 *lnB_ptr,
                                 int lnM) {
  const int thridx = threadIdx.x + threadIdx.y * blockDim.x;
  const int ctaIdx = blockIdx.x + blockIdx.y * gridDim.x;

  using gemmTileM = _128;
  using gemmTileN = _128;
  using gemmTileK = _32;
  using gemmPipe = _4;
  int mtileM = 32;
  int mtileSize = mtileM * gridDim.y;
  int mtileIdx = ctaIdx / mtileSize;
  int tileIdx = ctaIdx % mtileSize;
  int mtileMSize = min(mtileM, gridDim.x - mtileIdx * mtileM);
  int mIdx = mtileIdx * mtileM + tileIdx % mtileMSize;
  int nIdx = tileIdx / mtileMSize;
  auto ctaCoord = make_coord(mIdx, nIdx, _);
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
  using AsyncCopy =
      Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, fp16>;

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

  fp16 *shmem_fp16 = (fp16 *)shmem_ptr;
  fp16 *sA = shmem_fp16;
  shmem_fp16 += cosize_v<decltype(layoutASmem)>;
  fp16 *sB = shmem_fp16;
  shmem_fp16 += cosize_v<decltype(layoutBSmem)>;

  fp16 *sC = (fp16 *)shmem_ptr; // sA is out of scope when sC is alive
  Tensor tASmem = make_tensor(make_smem_ptr(sA), layoutASmem);
  Tensor tBSmem = make_tensor(make_smem_ptr(sB), layoutBSmem);
  Tensor tCSmem = make_tensor(make_smem_ptr(sC), layoutCSmem);

  auto copyA =
      make_tiled_copy(AsyncCopy{}, make_layout(Shape<_32, _4>{}, LayoutRight{}),
                      make_layout(Shape<_1, _8>{}, LayoutRight{}));
  auto copyB = copyA;

  ThrCopy thrCopyA = copyA.get_slice(thridx);
  Tensor copyASrc = thrCopyA.partition_S(tAGmem);
  Tensor copyADst = thrCopyA.partition_D(tASmem);
  ThrCopy thrCopyB = copyB.get_slice(thridx);
  Tensor copyBSrc = thrCopyB.partition_S(tBGmem);
  Tensor copyBDst = thrCopyB.partition_D(tBSmem);

  auto blockMMA = make_tiled_mma(
      MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>{},
      make_layout(Shape<_2, _2>{}, LayoutRight{}),
      Tile<decltype(get<0>(gemmTiler)), decltype(get<1>(gemmTiler)), _16>{});
  ThrMMA thrMMA = blockMMA.get_slice(thridx);
  Tensor mmaASmem = thrMMA.partition_A(tASmem);
  Tensor mmaBSmem = thrMMA.partition_B(tBSmem);
  Tensor mmaCSmem = thrMMA.partition_C(tCSmem);

  Tensor mmaAReg = thrMMA.make_fragment_A(mmaASmem(_, _, _, 0));
  Tensor mmaBReg = thrMMA.make_fragment_B(mmaBSmem(_, _, _, 0));
  Tensor mmaCReg = thrMMA.make_fragment_C(mmaCSmem);

  clear(mmaCReg);
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
  int copyIdx = 0;

  const int cta4Elemwise = gridDim.x * gridDim.y / 4;
  constexpr int elemwiseGranularity = 1024;
  int blockWorkload = lnM / cta4Elemwise;
  blockWorkload = ALIGN_UP(blockWorkload, elemwiseGranularity);
  int blockStart = blockWorkload * (ctaIdx - cta4Elemwise * 2);
  int blockEnd = blockStart + blockWorkload;
  blockStart = min(blockStart, lnM);
  blockEnd = min(blockEnd, lnM);
  int perStageWorkload = (blockEnd - blockStart) / kTileCount;
  perStageWorkload = ALIGN_UP(perStageWorkload, elemwiseGranularity);
  int kSiluTileCount = DIV_UP(blockEnd - blockStart, perStageWorkload);

  auto layoutSiluSmem = Layout<Shape<Int<elemwiseGranularity>, gemmPipe>>{};
  fp16 *siluA = shmem_fp16;
  shmem_fp16 += cosize_v<decltype(layoutSiluSmem)>;
  Tensor tSiluSmem = make_tensor(make_smem_ptr(siluA), layoutSiluSmem);
  Tensor tSiluAGmem =
      make_tensor(make_gmem_ptr(lnA_ptr + blockStart),
                  make_layout(make_shape(blockEnd - blockStart)));
  auto copySiluA =
      make_tiled_copy(AsyncCopy{}, Layout<Shape<_128>>{}, Layout<Shape<_8>>{});
  ThrCopy thrcopySiluA = copySiluA.get_slice(thridx);
  Tensor copySiluASrc = thrcopySiluA.partition_S(tSiluAGmem);
  Tensor copySiluADst = thrcopySiluA.partition_D(tSiluSmem);
  bool predSilu = 2 * cta4Elemwise <= ctaIdx && ctaIdx < 3 * cta4Elemwise;

  for (int pipeIdx = 0; pipeIdx < kTileCount + kMaxPipe - 1; pipeIdx++) {
    if (pipeIdx < kTileCount) {
      copy(copyA, copyASrc(_, _, _, pipeIdx), copyADst(_, _, _, copyIdx));
      copy(copyB, copyBSrc(_, _, _, pipeIdx), copyBDst(_, _, _, copyIdx));
    }
    if (pipeIdx < kSiluTileCount) {
      if (predSilu) {
        copy(copySiluA, copySiluASrc(_, pipeIdx), copySiluADst(_, 0, copyIdx));
      }
    }
    if (pipeIdx < kTileCount) {
      cp_async_fence();
    }
    int mmaIdx = copyIdx + 1;
    mmaIdx = mmaIdx == kMaxPipe ? 0 : mmaIdx;
    if (kMaxPipe - 1 <= pipeIdx) {
      cp_async_wait<kMaxPipe - 1>();
      __syncthreads();
    }
    if (kMaxPipe - 1 <= pipeIdx) {
      CUTE_UNROLL
      for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
        copy(ldmA, ldmASrc(_, _, blockIdx, mmaIdx), ldmADst(_, _, blockIdx));
        copy(ldmB, ldmBSrc(_, _, blockIdx, mmaIdx), ldmBDst(_, _, blockIdx));
        gemm(blockMMA, mmaAReg(_, _, blockIdx), mmaBReg(_, _, blockIdx),
             mmaCReg);
      }
    }
    if (kMaxPipe - 1 <= pipeIdx && pipeIdx < kSiluTileCount + kMaxPipe - 1) {
      // begin elemwise op
      int jobidx = pipeIdx - kMaxPipe + 1;
      if (predSilu) {
        for (int i = 0; i < elemwiseGranularity; i += 8 * 128) {
          fp16 *baseptr = &tSiluSmem(_, mmaIdx)[i];
          uint128_t val = ((uint128_t *)baseptr)[thridx];
          fp16 *ptr = (fp16 *)&val;
          CUTE_UNROLL
          for (int k = 0; k < 8; k++) {
            float tmp = float(ptr[k]);
            ptr[k] = __float2half(tmp / (1 + expf(-tmp)));
          }
          baseptr = lnB_ptr + blockStart + jobidx * elemwiseGranularity + i;
          ((uint128_t *)baseptr)[thridx] = val;
        }
      }
    }

    copyIdx += 1;
    copyIdx = copyIdx == kMaxPipe ? 0 : copyIdx;
  }

  Tensor mmaCRegFp16 = make_fragment_like<fp16>(mmaCReg.layout());
  using StMatrix = Copy_Atom<UniversalCopy<AlignedArray<fp16, 2>>, fp16>;
  using Copy8B = Copy_Atom<UniversalCopy<AlignedArray<fp16, 4>>, fp16>;
  auto stmC = make_tiled_copy_C(StMatrix{}, blockMMA);
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
    CUTE_UNROLL
    for (int i = 0; i < size(regFp32); i += 2) {
      __half2 packedHalf = __floats2half2_rn(regFp32[i], regFp32[i + 1]);
      regFp16[i] = packedHalf.x;
      regFp16[i + 1] = packedHalf.y;
    }
    copy(stmC, regFp16, smem);
  }
  __syncthreads();
  copy(copyC, copyCSrc, copyCDst);
}

void entry_cute_parallel_gemmrc_lnr(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                    fp16 *gemmC_ptr, int gemmM, int gemmN,
                                    int gemmK, fp16 *lnA_ptr, fp16 *lnB_ptr,
                                    int lnM, int lnN) {
  dim3 gridDim(gemmM / 128, gemmN / 128);
  dim3 blockDim(128, 1, 1);

  constexpr int SMEM_SIZE = (64 + 8) * 1024;
  cudaSafeCall(cudaFuncSetAttribute(
      parallel_kernels::kernel_cute_parallel_gemm_ln,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  std::string name = "cutlass_parallel_gemmrc_lnr";
  double latency = test_pipeline(
      [&]() {
        parallel_kernels::
            kernel_cute_parallel_gemm_ln<<<gridDim, blockDim, SMEM_SIZE>>>(
                gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK, lnA_ptr,
                lnB_ptr, lnM * lnN);
      },
      name);
  double tflops = get_tflops(gemmM, gemmN, gemmK, latency);
  printf("%s: %.2f TFLOPS\n", name.data(), tflops);
}

} // namespace parallel_kernels
