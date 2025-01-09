#include <functional>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "pybind11/pybind11.h"

#include "cuda_utils.h"
#include "cutlass_utils.h"
#include "gemm_utils.h"

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

namespace cutlass_kernels {

using fp16 = cutlass::half_t;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

void entry_cutlass_gemmrc(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k) {
  using gemmKernel = cutlass::gemm::device::Gemm<
      fp16, RowMajor, fp16, ColumnMajor, fp16, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;

  float alpha = 1.0;
  float beta = 0.0;

  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta});
  gemmKernel gemm_op;

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "%s: failed to initialize gemm op\n", __PRETTY_FUNCTION__);
    return;
  }

  gemm_op(args);
}

static void *get_the_buffer(size_t size) {
  static size_t cur = 0;
  static void *buf = nullptr;
  if (cur < size) {
    cur = size;
    if (buf) {
      cudaSafeCall(cudaFree(buf));
    }
    cudaSafeCall(cudaMalloc(&buf, size));
  }
  return buf;
}

void entry_cutlass_gemmrc_spec(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k,
                               std::vector<int> shape_threadblock,
                               std::vector<int> shape_warp) {
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      fp16, 128 / (8 * sizeof(fp16)), float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  // using Epilogue = cutlass::epilogue::thread::Identity<Half>;
  constexpr int numStages = 4;

  float alpha(1.0);
  float beta(0.0);

  if (shape_threadblock.size() == 0) {
    shape_threadblock = {128, 128};
  }
  if (shape_threadblock.size() != 2) {
    return;
  }
  if (shape_warp.size() == 0) {
    shape_warp = {64, 64};
  }
  if (shape_warp.size() != 2) {
    return;
  }
  int tm = shape_threadblock[0], tn = shape_threadblock[1];
  int wm = shape_warp[0], wn = shape_warp[1];

#define GENERATE_GEMM(_tm, _tn, _tk, _wm, _wn, _wk)                            \
  if (_tm == tm && _tn == tn && _wm == wm && _wn == wn) {                      \
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<_tm, _tn, _tk>;       \
    using ShapeMMAWarp = cutlass::gemm::GemmShape<_wm, _wn, _wk>;              \
    using gemmKernel = cutlass::gemm::device::Gemm<                            \
        fp16, RowMajor, fp16, ColumnMajor, fp16, RowMajor, float,              \
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,                   \
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, Epilogue,               \
        SwizzleThreadBlock, numStages>;                                        \
    gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},      \
                               {alpha, beta}, 1);                              \
    gemmKernel gemm_op;                                                        \
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {            \
      return;                                                                  \
    }                                                                          \
    size_t workspace_size = gemmKernel::get_workspace_size(args);              \
    auto workspace = get_the_buffer(workspace_size);                           \
    cutlassSafeCall(gemm_op(args, workspace));                                 \
  }

  GENERATE_GEMM(128, 128, 32, 64, 64, 32);
  GENERATE_GEMM(128, 256, 32, 64, 64, 32);
  GENERATE_GEMM(256, 128, 32, 64, 64, 32);

} // namespace cutlass_kernels

} // namespace cutlass_kernels
