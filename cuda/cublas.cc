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

void _cublas_gemm_nt(at::Tensor a, at::Tensor b, at::Tensor c) {
  checkTensor(a);
  checkTensor(b);
  checkTensor(c);

  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  checkIntEqual(a.size(1), b.size(1));
  checkIntEqual(m, c.size(0));
  checkIntEqual(n, c.size(1));

  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));
  __half *A = (__half *)a.data_ptr();
  __half *B = (__half *)b.data_ptr();
  __half *C = (__half *)c.data_ptr();

  std::string name =
      std::string("cublas::Hgemm_nt_{") + std::string(a.dtype().name()) + "," +
      std::string(b.dtype().name()) + "," + std::string(c.dtype().name()) + "}";

  if (tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
      tensorTypeIs<at::Half>(c)) {
    test_pipeline(
        [&]() {
          const __half alpha = __float2half(1.0);
          const __half beta = __float2half(0.0);
          cublasSafeCall(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                     &alpha, B, k, A, k, &beta, C, n));
        },
        name);
  } else {
    fprintf(stderr, "unsupported data type\n");
  }
  cublasSafeCall(cublasDestroy(handle));
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

void _cublas_gemmex_nt_compf32(at::Tensor a, at::Tensor b, at::Tensor c) {
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

  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));
  std::string name = std::string("cublas::GemmEx_nt_compf32_{") +
                     std::string(a.dtype().name()) + "," +
                     std::string(b.dtype().name()) + "," +
                     std::string(c.dtype().name()) + "}";
  double latency = test_pipeline(
      [&]() {
        float alpha = 1.0f, beta = 0.0f;
        cublasSafeCall(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                    &alpha, B, BType, k, A, AType, k, &beta, C,
                                    CType, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
      },
      name);
  cublasSafeCall(cublasDestroy(handle));
  double tflops = get_tflops(m, n, k, latency);
  printf("%s: %.2f TFLOPS\n", name.data(), tflops);
}

void register_cublas(pybind11::module &mod_perf, pybind11::module &mod_run) {
  mod_perf.def("cublas_gemm_nt", &_cublas_gemm_nt, "cublas_gemm_nt");
  mod_perf.def("cublas_gemmex_nt", &_cublas_gemmex_nt_compf32,
               "cublas_gemmex_nt");
}