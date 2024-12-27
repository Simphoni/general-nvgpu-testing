#include <cassert>

#include <cutlass/half.h>

#include "torch_utils.h"

using fp16 = cutlass::half_t;

namespace cutlass_kernels {

void entry_cutlass_gemmrc(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k);
void entry_cutlass_gemmrc_spec(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k);
void entry_cutlass_gemmrc_splitk(fp16 *A, fp16 *B, fp16 *C, int m, int n, int k,
                                 int split_k_slices);

} // namespace cutlass_kernels

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

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    cutlass_kernels::entry_cutlass_gemmrc(
        reinterpret_cast<fp16 *>(a.data_ptr()),
        reinterpret_cast<fp16 *>(b.data_ptr()),
        reinterpret_cast<fp16 *>(c.data_ptr()), m, n, k);
  } else {
    fprintf(stderr, "%s: unsupported tensor type\n", __PRETTY_FUNCTION__);
  }
}

void _cutlass_gemmrc_spec(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    cutlass_kernels::entry_cutlass_gemmrc_spec(
        reinterpret_cast<fp16 *>(a.data_ptr()),
        reinterpret_cast<fp16 *>(b.data_ptr()),
        reinterpret_cast<fp16 *>(c.data_ptr()), m, n, k);
  } else {
    fprintf(stderr, "%s: unsupported tensor type\n", __PRETTY_FUNCTION__);
  }
}

void _cutlass_gemmrc_splitk(at::Tensor a, at::Tensor b, at::Tensor c,
                            int split_k_slices) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    cutlass_kernels::entry_cutlass_gemmrc_splitk(
        reinterpret_cast<fp16 *>(a.data_ptr()),
        reinterpret_cast<fp16 *>(b.data_ptr()),
        reinterpret_cast<fp16 *>(c.data_ptr()), m, n, k, split_k_slices);
  } else {
    fprintf(stderr, "%s: unsupported tensor type\n", __PRETTY_FUNCTION__);
  }
}

void register_cutlass(pybind11::module &mod) {
  mod.def("cutlass_gemmrc_naive", &_cutlass_gemmrc);
  mod.def("cutlass_gemmrc_spec", &_cutlass_gemmrc_spec);
  mod.def("cutlass_gemmrc_splitk", &_cutlass_gemmrc_splitk, py::arg("a"),
          py::arg("b"), py::arg("c"), py::arg("split_k_slices") = 0);
}