
inline double get_tflops(int m, int n, int k, double latency_ms) {
  return 2.0 * m * n * k / latency_ms / 1e9;
}

int get_default_nrep();