#include <chrono>
#include <cstdio>
#include <functional>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include "torch_utils.h"

namespace ch = std::chrono;

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
}

#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, const char *file,
                             const int line) {
  if (CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr, "cublasSafeCall() failed at %s:%i : %d\n", file, line, err);
    exit(-1);
  }
}

void log_cublas_info() {
  static bool is_logged = false;
  if (is_logged) {
    return;
  }
  is_logged = true;
  // print cublas version info
  int major = 0, minor = 0, patch = 0;
  cublasSafeCall(cublasGetProperty(MAJOR_VERSION, &major));
  cublasSafeCall(cublasGetProperty(MINOR_VERSION, &minor));
  cublasSafeCall(cublasGetProperty(PATCH_LEVEL, &patch));
  fprintf(stderr, "CUBLAS version: %d.%d.%d\n", major, minor, patch);
}

void test_pipeline(std::function<void()> func, const std::string &name,
                   int repeat = 64) {
  fprintf(stderr, "%s: test pipeline running\n", name.data());
  for (int i = 0; i < repeat; i++) {
    func();
  }
  cudaSafeCall(cudaDeviceSynchronize());
  auto tic = ch::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    func();
  }
  cudaSafeCall(cudaDeviceSynchronize());
  auto toc = ch::high_resolution_clock::now();
  double duration =
      ch::duration_cast<ch::microseconds>(toc - tic).count() / 1000.0 / repeat;
  fprintf(stderr, "%s: %lf ms\n", name.data(), duration);
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
      std::string("cublas::Hgemm_nt_") + std::string(a.dtype().name()) + "_" +
      std::string(b.dtype().name()) + "_" + std::string(c.dtype().name());

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
  fflush(stderr);
  cublasSafeCall(cublasDestroy(handle));
}

inline cudaDataType get_cuda_data_type(at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Half:
    return CUDA_R_16F;
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
  assert(tensorTypeIs<at::Half>(a) && tensorTypeIs<at::Half>(b) &&
         tensorTypeIs<at::Half>(c));
  auto AType = get_tensor_data_type(a);
  auto BType = get_tensor_data_type(b);
  auto CType = get_tensor_data_type(c);
  void *A = a.data_ptr();
  void *B = b.data_ptr();
  void *C = c.data_ptr();

  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));
  std::string name = std::string("cublas::GemmEx_nt_compf32_") +
                     std::string(a.dtype().name()) + "_" +
                     std::string(b.dtype().name()) + "_" +
                     std::string(c.dtype().name());
  test_pipeline(
      [&]() {
        float alpha = 1.0f, beta = 0.0f;
        cublasSafeCall(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                    &alpha, B, BType, k, A, AType, k, &beta, C,
                                    CType, n, CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      },
      name);
  cublasSafeCall(cublasDestroy(handle));
}

// the exported functions should be named with a leading underscore
PYBIND11_MODULE(INTERFACE_NAME, m) {
  log_cublas_info();
  m.def("cublas_gemm_nt", &_cublas_gemm_nt, "cublas_gemm_nt");
  m.def("cublas_gemmex_nt", &_cublas_gemmex_nt_compf32, "cublas_gemmex_nt");
}