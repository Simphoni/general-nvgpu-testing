#include "cute/tensor.hpp"
#include "pybind11/pybind11.h"

#include "cuda_utils.h"
#include "gemm_utils.h"
#include "torch_utils.h"

using fp16 = cute::half_t;

namespace parallel_kernels {

using namespace cute;

extern __shared__ uint8_t shmem_ptr[];

__global__ void __launch_bounds__(128)
    cute_parallel_gemm_ln(fp16 *gemmA_ptr, fp16 *gemmB_ptr,
                          fp16 *gemmC_ptr, // fp16 *lnA, fp16 *lnB,
                          int gemmM, int gemmN, int gemmK) {
  using gemmTileM = _128;
  using gemmTileN = _128;
  using gemmTileK = _32;
  using gemmPipe = _4;
  const int thridx = threadIdx.x + threadIdx.y * blockDim.x;
  auto ctaCoord = make_coord(blockIdx.x, blockIdx.y, _);
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
  using GemmCopyIntrin16B = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;

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
  // [cosize_v<decltype(layoutASmem)>];
  fp16 *sA = (fp16 *)shmem_ptr;
  // [cosize_v<decltype(layoutBSmem)>];
  fp16 *sB = sA + cosize_v<decltype(layoutASmem)>;
  Tensor tASmem = make_tensor(make_smem_ptr(sA), layoutASmem);
  Tensor tBSmem = make_tensor(make_smem_ptr(sB), layoutBSmem);

  auto copyA =
      make_tiled_copy(Copy_Atom<Copy_Traits<GemmCopyIntrin16B>, fp16>{},
                      make_layout(Shape<_32, _4>{}, LayoutRight{}),
                      make_layout(Shape<_1, _8>{}, LayoutRight{}));
  auto copyB = copyA;

  // cA for [copy]TiledTensorA
  // (COPY,COPY_M,COPY_K,k_tile_count)
  ThrCopy thrCopyA = copyA.get_slice(thridx);
  Tensor copyASrc = thrCopyA.partition_S(tAGmem);
  Tensor copyADst = thrCopyA.partition_D(tASmem);
  ThrCopy thrCopyB = copyB.get_slice(thridx);
  Tensor copyBSrc = thrCopyB.partition_S(tBGmem);
  Tensor copyBDst = thrCopyB.partition_D(tBSmem);

  auto blockMMA = make_tiled_mma(
      MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>{},
      make_layout(Shape<_2, _2>{},
                  LayoutRight{}), // thr_layout tiles the work onto all warps
      Tile<decltype(get<0>(gemmTiler)), decltype(get<1>(gemmTiler)), _16>{});
  ThrMMA thrMMA = blockMMA.get_slice(thridx);
  // partition_A expect (M, K, ...)
  Tensor mmaASmem = thrMMA.partition_A(tASmem);
  // partition_B expect (N, K, ...)
  Tensor mmaBSmem = thrMMA.partition_B(tBSmem);
  // partition_C expect (M, N, ...)
  Tensor mmaCGmem = thrMMA.partition_C(tCGmem);

  Tensor mmaAReg = thrMMA.make_fragment_A(mmaASmem(_, _, _, 0));
  Tensor mmaBReg = thrMMA.make_fragment_B(mmaBSmem(_, _, _, 0));
  Tensor mmaCReg = thrMMA.make_fragment_C(mmaCGmem);

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
    copy(copyA, copyASrc(_, _, _, pipeIdx), copyADst(_, _, _, loadIdx));
    copy(copyB, copyBSrc(_, _, _, pipeIdx), copyBDst(_, _, _, loadIdx));
    cp_async_fence();
    int mmaIdx = 0;
    cp_async_wait<0>();
    // loading from shared requires barrier
    __syncthreads();
    CUTE_UNROLL
    for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
      copy(ldmA, ldmASrc(_, _, blockIdx, mmaIdx), ldmADst(_, _, blockIdx));
      copy(ldmB, ldmBSrc(_, _, blockIdx, mmaIdx), ldmBDst(_, _, blockIdx));
      gemm(blockMMA, mmaAReg(_, _, blockIdx), mmaBReg(_, _, blockIdx), mmaCReg);
    }
    loadIdx += 1;
    loadIdx = loadIdx == kMaxPipe ? 0 : loadIdx;
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
      // loading from shared requires barrier
      __syncthreads();
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
  for (int j = 0; j < size<2>(mmaCReg); j++) {
    auto regFp16 = mmaCRegFp16(_, _, j);
    auto regFp32 = mmaCReg(_, _, j);
    auto gmem = mmaCGmem(_, _, j);
    CUTE_UNROLL
    for (int i = 0; i < size(regFp32); ++i) {
      regFp16[i] = __float2half(regFp32[i]);
    }
    copy(regFp16, gmem);
  }
}

} // namespace parallel_kernels

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

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
  dim3 blockDim(128, 1, 1);
  fp16 *gemmA_ptr = reinterpret_cast<fp16 *>(gemmA.data_ptr());
  fp16 *gemmB_ptr = reinterpret_cast<fp16 *>(gemmB.data_ptr());
  fp16 *gemmC_ptr = reinterpret_cast<fp16 *>(gemmC.data_ptr());

  constexpr int SMEM_SIZE = 64 * 1024;
  cudaSafeCall(cudaFuncSetAttribute(parallel_kernels::cute_parallel_gemm_ln,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    SMEM_SIZE));
  test_pipeline(
      [&]() {
        parallel_kernels::
            cute_parallel_gemm_ln<<<gridDim, blockDim, SMEM_SIZE>>>(
                gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK);
      },
      "cutlass_parallel_gemmrc_layernorm");
}

// void _cute_play() {
//   using namespace cute;
//   using mma_op = SM80_16x8x16_F16F16F16F16_TN;

//   using mma_traits = MMA_Traits<mma_op>;
//   using mma_atom = MMA_Atom<mma_traits>;

//   static constexpr int kMmaEURepeatM = 2;
//   static constexpr int kMmaEURepeatN = 2;
//   static constexpr int kMmaEURepeatK = 1;

//   using mma_atom_shape = mma_traits::Shape_MNK;
//   static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
//   static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
//   static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
//   using MMA_EU_RepeatT = decltype(make_layout(make_shape(
//       Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
//   using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
//   using SmemLayoutAtomC = decltype(composition(
//       Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{},
//       Int<kMmaPN>{}),
//                                       make_stride(Int<kMmaPN>{},
//                                       Int<1>{}))));
//   using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{},
//   MMA_P_T{})); print_latex(SmemLayoutAtomC{}); using s2r_copy_op =
//   SM75_U32x4_LDSM_N; using s2r_copy_traits = Copy_Traits<s2r_copy_op>; using
//   s2r_copy_atom = Copy_Atom<s2r_copy_traits, fp16>; MMA tiled_mma; auto
//   s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom{}, tiled_mma);
//   print_latex(s2r_tiled_copy_a);
// }

void register_cutlass_parallel(pybind11::module &m) {
  m.def("cutlass_parallel_gemmrc_layernorm",
        &_cutlass_parallel_gemmrc_layernorm);
  // m.def("cute_play", &_cute_play);
}
