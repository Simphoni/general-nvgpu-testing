#pragma once

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
