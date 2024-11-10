#include "cute/tensor.hpp"
#include "cute/tensor_impl.hpp"
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
  using gemmPipe = _5;
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

  // tX for tensorX
  Tensor tAGmem = local_tile(tensorA, gemmTiler, ctaCoord, Step<_1, X, _1>{});
  // when masked, ctaCoord become (blockIdx.x, _), selecting all the column
  Tensor tBGmem = local_tile(tensorB, gemmTiler, ctaCoord, Step<X, _1, _1>{});
  Tensor tCGmem = local_tile(tensorC, gemmTiler, ctaCoord, Step<_1, _1, X>{});

  // auto sA_atom =
  //     make_layout(make_shape(gemmTileM{}, gemmTileK{}), LayoutRight{});
  // auto sAshape =
  //     tile_to_shape(sA_atom, make_shape(gemmTileM{}, gemmTileK{},
  //     gemmPipe{}));

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
  Tensor cAGmem = thrCopyA.partition_S(tAGmem);
  Tensor cASmem = thrCopyA.partition_D(tASmem);
  ThrCopy thrCopyB = copyB.get_slice(thridx);
  Tensor cBGmem = thrCopyB.partition_S(tBGmem);
  Tensor cBSmem = thrCopyB.partition_D(tBSmem);

  auto blockMMA = make_tiled_mma(
      MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>{},
      make_layout(Shape<_2, _2>{},
                  LayoutRight{}), // thr_layout tiles the work onto all warps
      Tile<decltype(get<0>(gemmTiler)), decltype(get<1>(gemmTiler)), _16>{});
  ThrMMA thrMMA = blockMMA.get_slice(thridx);
  // mA for [mma]TiledTensorA
  // auto tmp1 = blockMMA.thrfrg_B(tBSmem);
  Tensor mASmem = thrMMA.partition_A(tASmem); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor mBSmem = thrMMA.partition_B(tBSmem);
  Tensor mCGmem = thrMMA.partition_C(tCGmem);

  // Tensor mAReg = make_tensor<fp16>(mASmem(_, _, 0, 0).shape());
  // Tensor mBReg = make_tensor<fp16>(mBSmem(_, _, 0, 0).shape());
  Tensor mAReg = thrMMA.make_fragment_A(mASmem(_, _, _, 0));
  Tensor mBReg = thrMMA.make_fragment_B(mBSmem(_, _, _, 0));
  Tensor mCReg = thrMMA.make_fragment_C(mCGmem);

  clear(mCReg);

  constexpr int kMaxPipe = size<3>(mASmem);
  int kTileCount = size<3>(cAGmem);
  int kMaxBlock = size<2>(mASmem);
  int loadIdx = 0;
  for (int pipeIdx = 0; pipeIdx < kTileCount + kMaxPipe - 1; pipeIdx++) {
    if (pipeIdx < kTileCount) {
      copy(copyA, cAGmem(_, _, _, pipeIdx), cASmem(_, _, _, loadIdx));
      copy(copyB, cBGmem(_, _, _, pipeIdx), cBSmem(_, _, _, loadIdx));
    }
    if (kMaxPipe - 1 <= pipeIdx) {
      int mmaIdx = loadIdx + 1;
      mmaIdx = mmaIdx == kMaxPipe ? 0 : mmaIdx;
      cp_async_wait<kMaxPipe - 1>();
      __syncthreads();
      CUTE_UNROLL
      for (int blockIdx = 0; blockIdx < kMaxBlock; blockIdx++) {
        copy(mASmem(_, _, blockIdx, mmaIdx), mAReg(_, _, blockIdx));
        copy(mBSmem(_, _, blockIdx, mmaIdx), mBReg(_, _, blockIdx));
        gemm(blockMMA, mAReg(_, _, blockIdx), mBReg(_, _, blockIdx), mCReg);
      }
    }
    loadIdx += 1;
    loadIdx = loadIdx == kMaxPipe ? 0 : loadIdx;
  }
  Tensor mCRegFp16 = make_fragment_like<fp16>(mCReg.layout());
  CUTE_UNROLL
  for (int i = 0; i < size(mCReg); ++i) {
    mCRegFp16[i] = __float2half(mCReg[i]);
  }
  copy(mCRegFp16, mCGmem);
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
  cudaSafeCall(cudaFuncSetAttribute(parallel_kernels::cute_parallel_gemm_ln,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    80 * 1024));
  test_pipeline(
      [&]() {
        parallel_kernels::
            cute_parallel_gemm_ln<<<gridDim, blockDim, 80 * 1024>>>(
                gemmA_ptr, gemmB_ptr, gemmC_ptr, gemmM, gemmN, gemmK);
      },
      "cutlass_parallel_gemmrc_layernorm");
}

void _cute_play() {
  using namespace cute;
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  print_latex(SmemLayoutAtomC{});
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, fp16>;
  MMA tiled_mma;
  auto s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom{}, tiled_mma);
  print_latex(s2r_tiled_copy_a);
}

void register_cutlass_parallel(pybind11::module &m) {
  m.def("cutlass_parallel_gemmrc_layernorm",
        &_cutlass_parallel_gemmrc_layernorm);
  m.def("cute_play", &_cute_play);
}
