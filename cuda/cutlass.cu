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

void _cutlass_gemmrc(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));

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

  gemmKernel::Arguments args({m, n, k}, {A, k}, {B, k}, {C, n}, {C, n},
                             {alpha, beta});
  gemmKernel gemm_op;

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "%s: failed to initialize gemm op\n", __PRETTY_FUNCTION__);
    return;
  }

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    gemm_op(args);
  } else {
    fprintf(stderr, "unsupported data type\n");
  }
}

void register_cutlass(pybind11::module &mod) {
  mod.def("cutlass_gemmrc_naive", &_cutlass_gemmrc);
}