#include <set>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

#include "cuda_utils.h"
#include "cutlass_utils.h"

using fp16 = cutlass::half_t;

class Logger {
private:
  std::set<std::string> logs;

public:
  void info(const std::string msg) {
    if (logs.find(msg) == logs.end()) {
      logs.insert(msg);
      fprintf(stderr, "[INFO] %s\n", msg.c_str());
    }
  }
};
static Logger _logger;

namespace cutlass_kernels {

static constexpr float kMaxBlock = 108;

// this function only holds one buffer
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

static int get_slice_heuristic(int maxsplit, int occupancy) {
  int split_k_slices = 1;
  if (occupancy < kMaxBlock) {
    split_k_slices = (kMaxBlock - 1) / occupancy + 1;
    float waves = split_k_slices * occupancy / kMaxBlock;
    float rem = waves - int(waves);
    while (rem < 0.5) {
      split_k_slices++;
      waves = split_k_slices * occupancy / kMaxBlock;
      rem = waves - int(waves);
    }
  }
  return split_k_slices;
}

void entry_cutlass_gemmrc_splitk(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k,
                                 int _split_k_slices) {
  using ElementAccumulator = float;
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using gemmKernel = cutlass::gemm::device::GemmSplitKParallel<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch>;

  int split_k_slices = 1;
  int maxsplit = k / ShapeMMAThreadBlock::kK;

  if (_split_k_slices > 0 && _split_k_slices < maxsplit) {
    split_k_slices = _split_k_slices;
  } else {
    int occupancy = m / ShapeMMAThreadBlock::kM * n / ShapeMMAThreadBlock::kN;
    split_k_slices = get_slice_heuristic(maxsplit, occupancy);
    split_k_slices = std::min(split_k_slices, maxsplit);
    split_k_slices = std::max(split_k_slices, 1);
    std::string prob_shape = "[" + std::to_string(m) + "," + std::to_string(n) +
                             "," + std::to_string(k) + "]";
    _logger.info(std::string(__PRETTY_FUNCTION__) + ": " + prob_shape +
                 " split_k(" + std::to_string(split_k_slices) + "), CTAs(" +
                 std::to_string(occupancy * split_k_slices) + "), waves(" +
                 std::to_string(occupancy * split_k_slices / kMaxBlock) + ")");
  }

  gemmKernel gemm_op;
  ElementAccumulator alpha = (ElementAccumulator)1.0,
                     beta = (ElementAccumulator)0.0;
  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta}, split_k_slices);
  auto workspace_size = gemmKernel::get_workspace_size(args);
  auto workspace = get_the_buffer(workspace_size);
  cutlassSafeCall(gemm_op.can_implement(args));
  cutlassSafeCall(gemm_op(args, workspace));
}

void entry_cutlass_gemmrc_splitk_spec(fp16 *A, fp16 *B, fp16 *C, int m, int n,
                                      int k, int _split_k_slices) {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,

      ElementAccumulator, ElementComputeEpilogue>;

  using gemmKernel = cutlass::gemm::device::GemmSplitKParallel<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp>;

  int split_k_slices = 1;
  int maxsplit = k / ShapeMMAThreadBlock::kK;

  if (_split_k_slices > 0 && _split_k_slices < split_k_slices) {
    split_k_slices = _split_k_slices;
  } else {
    int occupancy = m / ShapeMMAThreadBlock::kM * n / ShapeMMAThreadBlock::kN;
    split_k_slices = get_slice_heuristic(maxsplit, occupancy);
    split_k_slices = std::min(split_k_slices, maxsplit);
    split_k_slices = std::max(split_k_slices, 1);
    std::string prob_shape = "[" + std::to_string(m) + "," + std::to_string(n) +
                             "," + std::to_string(k) + "]";
    _logger.info(std::string(__PRETTY_FUNCTION__) + ": " + prob_shape +
                 " split_k(" + std::to_string(split_k_slices) + "), CTAs(" +
                 std::to_string(occupancy * split_k_slices) + "), waves(" +
                 std::to_string(occupancy * split_k_slices / kMaxBlock) + ")");
  }

  gemmKernel gemm_op;
  ElementAccumulator alpha = (ElementAccumulator)1.0,
                     beta = (ElementAccumulator)0.0;
  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta}, split_k_slices);
  auto workspace_size = gemmKernel::get_workspace_size(args);
  auto workspace = get_the_buffer(workspace_size);
  cutlassSafeCall(gemm_op.can_implement(args));
  cutlassSafeCall(gemm_op(args, workspace));
}

} // namespace cutlass_kernels