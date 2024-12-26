#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

#include "pybind11/pybind11.h"

#include "cuda_utils.h"
#include "gemm_utils.h"
#include "torch_utils.h"

double test_pipeline(std::function<void()> func, const std::string &name,
                     int repeat = -1) {
  constexpr int sleep_ms = 100;
  if (repeat == -1) {
    repeat = get_default_nrep();
  }
  fprintf(stderr,
          "%s: test pipeline running: warmup=%d, sleep=%d ms, nrep=%d\n",
          name.data(), repeat, sleep_ms, repeat);
  for (int i = 0; i < repeat; i++) {
    func();
  }
  cudaSafeCall(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  cudaEvent_t tic, toc;
  cudaSafeCall(cudaEventCreate(&tic));
  cudaSafeCall(cudaEventCreate(&toc));
  cudaSafeCall(cudaEventRecord(tic));
  for (int i = 0; i < repeat; i++) {
    func();
  }
  cudaSafeCall(cudaEventRecord(toc));
  cudaSafeCall(cudaEventSynchronize(toc));
  float time;
  cudaSafeCall(cudaEventElapsedTime(&time, tic, toc));
  double duration = time / repeat;
  fprintf(stderr, "%s: %lf ms\n", name.data(), duration);
  return duration;
}

cublasHandle_t get_cublas_handle();

void _cublas_hgemm_rc(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));

  cublasHandle_t handle = get_cublas_handle();
  __half *A = (__half *)a.data_ptr();
  __half *B = (__half *)b.data_ptr();
  __half *C = (__half *)c.data_ptr();

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    const __half alpha = __float2half(1.0);
    const __half beta = __float2half(0.0);
    cublasSafeCall(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                               &alpha, B, k, A, k, &beta, C, n));

  } else {
    fprintf(stderr, "unsupported data type\n");
  }
}

inline cudaDataType get_cuda_data_type(at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Half:
    return CUDA_R_16F;
  case at::ScalarType::BFloat16:
    return CUDA_R_16BF;
  case at::ScalarType::Float:
    return CUDA_R_32F;
  case at::ScalarType::Double:
    return CUDA_R_64F;
  default:
    fprintf(stderr, "cublas::unsupported data type\n");
    throw;
  }
}

inline cudaDataType get_tensor_data_type(at::Tensor a) {
  return get_cuda_data_type(a.scalar_type());
}

void _cublas_gemmex_rc_compf32(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));
  bool is_fp16 = (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
                  tensorTypeIs<at::Half>(c));
  bool is_bf16 =
      (tensorTypeIs<at::BFloat16>(a) && tensorTypeIs<at::BFloat16>(b) &&
       tensorTypeIs<at::BFloat16>(c));
  assert(is_fp16 || is_bf16);
  auto AType = get_tensor_data_type(a);
  auto BType = get_tensor_data_type(b);
  auto CType = get_tensor_data_type(c);
  void *A = a.data_ptr();
  void *B = b.data_ptr();
  void *C = c.data_ptr();

  cublasHandle_t handle = get_cublas_handle();
  float alpha = 1.0f, beta = 0.0f;
  cublasSafeCall(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              B, BType, k, A, AType, k, &beta, C, CType, n,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

void register_cublas(pybind11::module &mod) {
  mod.def("cublas_hgemmrc", &_cublas_hgemm_rc);
  mod.def("cublas_gemmexrc", &_cublas_gemmex_rc_compf32);
}