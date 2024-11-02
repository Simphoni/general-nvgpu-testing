#include <functional>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/version.h"
#include "pybind11/pybind11.h"

#include "gemm_utils.h"
#include "torch_utils.h"

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1);

void _cutlass_gemm_nt_naive(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));
  std::string name = std::string("cutlass::gemm_nt_{comp=fp32,") +
                     std::string(a.dtype().name()) + "," +
                     std::string(b.dtype().name()) + "," +
                     std::string(c.dtype().name()) + "}";

  using Half = cutlass::half_t;
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using gemmKernel = cutlass::gemm::device::Gemm<
      Half, RowMajor, Half, ColumnMajor, Half, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;

  Half *A = (Half *)a.data_ptr();
  Half *B = (Half *)b.data_ptr();
  Half *C = (Half *)c.data_ptr();

  float alpha = 1.0;
  float beta = 0.0;
  // using TensorRef = cutlass::TensorRef<Half, RowMajor>;
  // gemmKernel::Arguments args(
  //     cutlass::gemm::GemmCoord(m, n, k), TensorRef(A, RowMajor(k)),
  //     TensorRef(B, RowMajor(k)), TensorRef(C, RowMajor(n)),
  //     TensorRef(C, RowMajor(n)));

  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta});
  gemmKernel gemm_op;

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "%s: failed to initialize gemm op\n", name.c_str());
    return;
  }

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    double latency = test_pipeline([&]() { gemm_op(args); }, name);
    fprintf(stderr, "---> TFLOPS: %.4f\n", get_tflops(m, n, k, latency));
  } else {
    fprintf(stderr, "unsupported data type\n");
  }
}

void register_cutlass(pybind11::module &mod_perf, pybind11::module &mod_run) {
  mod_perf.def("cutlass_gemm_nt_naive", &_cutlass_gemm_nt_naive);
}