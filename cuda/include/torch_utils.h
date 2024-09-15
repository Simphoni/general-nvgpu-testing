#include <string>
#include <torch/extension.h>

inline void checkTensorDevice(at::Tensor a, const std::string &name) {
  if (a.device().type() != c10::DeviceType::CUDA) {
    fprintf(stderr, "[ERROR] tensor %s expected MLU, got %s\n", name.data(),
            DeviceTypeName(a.device().type()).data());
    throw;
  }
}

inline void checkTensorContiguous(at::Tensor a, const std::string &name) {
  if (!a.is_contiguous()) {
    fprintf(stderr, "[ERROR] tensor %s not contiguous\n", name.data());
    throw;
  }
}

#define checkTensor(x)                                                         \
  __checkTensor(x, std::string(__PRETTY_FUNCTION__) + ":" + #x)

inline void __checkTensor(at::Tensor a, const std::string &name) {
  checkTensorDevice(a, name);
  checkTensorContiguous(a, name);
}

#define checkIntEqual(a, b)                                                    \
  __checkIntEqual((int64_t)a, (int64_t)b, __FILE__, __LINE__)

inline void __checkIntEqual(int64_t a, int64_t b, const char *file, int line) {
  if (a != b) {
    fprintf(stderr, "%s:%d [ERROR] checkEqual failed (%ld != %ld)\n", file,
            line, a, b);
    throw;
  }
}

inline bool checkSafelyConvertToInt32(int64_t a) {
  return a >= 0 && a <= INT_MAX;
}

template <typename T> inline bool tensorTypeIs(const at::Tensor &a) {
  return a.dtype().id() == caffe2::TypeIdentifier::Get<T>();
}