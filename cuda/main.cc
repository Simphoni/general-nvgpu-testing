#include <cutlass/version.h>

#include "cuda_utils.h"
#include "pybind11/pybind11.h"
#include "torch_utils.h"

void log_cublas_version() {
  int major = 0, minor = 0, patch = 0;
  cublasSafeCall(cublasGetProperty(MAJOR_VERSION, &major));
  cublasSafeCall(cublasGetProperty(MINOR_VERSION, &minor));
  cublasSafeCall(cublasGetProperty(PATCH_LEVEL, &patch));
  fprintf(stderr, "CUBLAS version: %d.%d.%d\n", major, minor, patch);
}

void log_cutlass_version() {
  fprintf(stderr, "CUTLASS version: %d.%d.%d\n", cutlass::getVersionMajor(),
          cutlass::getVersionMinor(), cutlass::getVersionPatch());
}

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle;
  static bool init = false;
  if (!init) {
    init = true;
    cublasSafeCall(cublasCreate(&handle));
    return handle;
  }
  return handle;
}

void register_cublas(pybind11::module &mod);
void register_cutlass(pybind11::module &mod);
void register_cute_kernels(pybind11::module &mod);

namespace {

int default_nrep{2};
void set_default_nrep(int nrep) {
  if (nrep <= 0 || nrep > 64) {
    fprintf(stderr, "nrep should be in [1, 64], got %d\n", nrep);
    return;
  }
  default_nrep = nrep;
}

} // namespace

int get_default_nrep() { return default_nrep; }

// the exported functions should be named with a leading underscore
PYBIND11_MODULE(INTERFACE_NAME, m) {
  log_cublas_version();
  log_cutlass_version();
  get_cublas_handle();
  // pybind11::module m_perf = m.def_submodule("perf");
  // m_perf.def("set_default_nrep", &set_default_nrep);
  register_cublas(m);
  register_cutlass(m);
  register_cute_kernels(m);
}