#include <functional>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/version.h"
#include "pybind11/pybind11.h"

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

void entry_cutlass_gemmrc_spec(fp16 *A, fp16 *B, fp16 *C, int m, int n,
                                int k) {
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  // using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>; // this is for SM_70
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      fp16, 128 / (8 * sizeof(fp16)), float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  // using Epilogue = cutlass::epilogue::thread::Identity<Half>;
  constexpr int numStages = 4;

  using gemmKernel = cutlass::gemm::device::Gemm<
      fp16, RowMajor, fp16, ColumnMajor, fp16, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      // fine-grain shape config
      ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, Epilogue,
      SwizzleThreadBlock, numStages>;

  float alpha(1.0);
  float beta(0.0);

  int split_k_slices = 1;
  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta}, split_k_slices);
  gemmKernel gemm_op;
  size_t workspace_size = gemmKernel::get_workspace_size(args);
  if (workspace_size) {
    fprintf(stderr, "%s: workspace size: %lu\n", __PRETTY_FUNCTION__,
            workspace_size);
    return;
  }
  gemm_op(args);
}

} // namespace cutlass_kernels
