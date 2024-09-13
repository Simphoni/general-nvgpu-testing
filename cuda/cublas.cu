#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "cublasSafeCall() failed at %s:%i : %d\n", file, line, err);
        exit(-1);
    }
}

#define curandSafeCall(err) __curandSafeCall(err, __FILE__, __LINE__)
inline void __curandSafeCall(curandStatus_t err, const char *file, const int line) {
    if (CURAND_STATUS_SUCCESS != err) {
        fprintf(stderr, "curandSafeCall() failed at %s:%i : %d\n", file, line, err);
        exit(-1);
    }
}

int main() {
    int n = 4096;
    int m = 4096;
    int k = 4096;
    
}