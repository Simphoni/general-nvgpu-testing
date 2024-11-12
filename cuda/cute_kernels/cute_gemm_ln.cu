// #include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>

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

constexpr int kThreadGroupSize = 32 * 4;

static __global__ void __launch_bounds__(kThreadGroupSize)
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
  int mtileM = 8;
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
  fp16 *sA = (fp16 *)shmem_ptr;
  fp16 *sB = sA + cosize_v<decltype(layoutASmem)>;
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
  int loadIdx = 0;

  int usePrologue = 1; // 0 or 1
  if (usePrologue) {
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

  for (int pipeIdx = usePrologue; pipeIdx < kTileCount + kMaxPipe - 1;
       pipeIdx++) {
    if (pipeIdx < kTileCount) {
      copy(copyA, copyASrc(_, _, _, pipeIdx), copyADst(_, _, _, loadIdx));
      copy(copyB, copyBSrc(_, _, _, pipeIdx), copyBDst(_, _, _, loadIdx));
      cp_async_fence();
    }
    // if (kMaxPipe - 1 <= pipeIdx - usePrologue) {
    // }
    if (kMaxPipe - 1 <= pipeIdx - usePrologue) {
      cp_async_wait<kMaxPipe - 1>();
      __syncthreads();
      int mmaIdx = loadIdx + 1;
      mmaIdx = mmaIdx == kMaxPipe ? 0 : mmaIdx;
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

  // int total_workers = gridDim.x * gridDim.y;
  // int start_idx = (ctaIdx * 128 + thridx - 128) * 2;
  // int stride = total_workers * 128 * 2;
  // int step = 0;
  // for (int i = start_idx; i < lnM; i += stride) {
  //   if (step <= gemmK / 32) {
  //     __syncthreads();
  //   }
  //   __half2 tmp = *reinterpret_cast<__half2 *>(&lnA_ptr[i]);
  //   __half2 tmp2;
  //   tmp2.x = tmp.x / (__half(1.0) + hexp(-tmp.x));
  //   tmp2.y = tmp.y / (__half(1.0) + hexp(-tmp.y));
  //   *reinterpret_cast<__half2 *>(&lnB_ptr[i]) = tmp2;
  //   step++;
  // }
  // while (step <= gemmK / 32) {
  //   __syncthreads();
  //   step++;
  // }
}

void entry_cute_parallel_gemmrc_lnr(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                                    fp16 *gemmC_ptr, int gemmM, int gemmN,
                                    int gemmK, fp16 *lnA_ptr, fp16 *lnB_ptr,
                                    int lnM, int lnN) {
  dim3 gridDim(gemmM / 128, gemmN / 128);
  dim3 blockDim(kThreadGroupSize, 1, 1);

  constexpr int SMEM_SIZE = 64 * 1024;
  cudaSafeCall(cudaFuncSetAttribute(
      parallel_kernels::kernel_cute_parallel_gemm_ln,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  std::string name = "cutlass_parallel_gemmrc";
  // int lnSize = lnM * lnN;
  // void *kernelArgs[] = {(void *)gemmA_ptr, (void *)gemmB_ptr, (void
  // *)gemmC_ptr,
  //                       (void *)&gemmM,    (void *)&gemmN,    (void *)&gemmK,
  //                       (void *)lnA_ptr,   (void *)lnB_ptr,   (void
  //                       *)&lnSize};
  double latency = test_pipeline(
      [&]() {
        // cudaLaunchCooperativeKernel(
        //     (void *)parallel_kernels::kernel_cute_parallel_gemm_ln, gridDim,
        //     blockDim, kernelArgs, SMEM_SIZE, nullptr);
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
