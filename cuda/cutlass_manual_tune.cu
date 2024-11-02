#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/gemm/device/gemm.h"

#include "gemm_utils.h"
#include "torch_utils.h"

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

void _cutlass_gemm_nt_manual_tune(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));
  std::string name = std::string("cutlass::gemm_nt_manual_tune_{comp=fp32,") +
                     std::string(a.dtype().name()) + "," +
                     std::string(b.dtype().name()) + "," +
                     std::string(c.dtype().name()) + "}";

  using Half = cutlass::half_t;
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  // using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>; // this is for SM_70
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Half, 128 / (8 * sizeof(Half)), float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  // using Epilogue = cutlass::epilogue::thread::Identity<Half>;
  constexpr int numStages = 4;

  using gemmKernel = cutlass::gemm::device::Gemm<
      Half, RowMajor, Half, ColumnMajor, Half, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      // fine-grain shape config
      ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, Epilogue,
      SwizzleThreadBlock, numStages>;

  Half *A = (Half *)a.data_ptr();
  Half *B = (Half *)b.data_ptr();
  Half *C = (Half *)c.data_ptr();

  float alpha(1.0);
  float beta(0.0);

  int split_k_slices = 1;
  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta}, split_k_slices);
  gemmKernel gemm_op;
  size_t workspace_size = gemmKernel::get_workspace_size(args);
  printf("%s: workspace size: %lu\n", name.c_str(), workspace_size);
  assert(workspace_size == 0);

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    double latency = test_pipeline([&]() { gemm_op(args); }, name);
    fprintf(stderr, "---> TFLOPS: %.4f\n", get_tflops(m, n, k, latency));
  } else {
    fprintf(stderr, "unsupported data type\n");
  }
}

void register_cutlass_manual(pybind11::module &mod_perf, pybind11::module &mod_run) {
  mod_perf.def("cutlass_gemm_nt_manual_tune", &_cutlass_gemm_nt_manual_tune);
} 