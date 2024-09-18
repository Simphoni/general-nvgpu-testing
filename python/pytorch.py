import torch
import time

# from gemm_bound import GPU_SPEC_LIST

print(torch.__version__)


NUM_RUNS = 64

STR_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    # "int4": torch.int4,
}


import acre


def test_gemm(n: int, k: int, m: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    a = torch.randn(n, k, dtype=dtype).cuda()
    b = torch.randn(m, k, dtype=dtype).cuda()
    c = torch.randn(n, m, dtype=dtype).cuda()

    torch.autograd.set_grad_enabled(False)
    print(f"[TEST] GEMM: n={n}, k={k}, m={m}, dtype={dtype}")

    # pytorch
    def run():
        torch.matmul(a, b.t(), out=c)

    for _ in range(NUM_RUNS):
        run()

    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(NUM_RUNS):
        run()
    torch.cuda.synchronize()
    toc = time.time()
    latency = (toc - tic) / NUM_RUNS

    print(f"PyTorch: {latency * 1e3} ms")

    d = torch.zeros_like(c)
    acre.cublas_gemmex_nt(a, b, d)
    torch.testing.assert_close(c, d, rtol=1e-3, atol=1e-3)
    acre.cublas_gemm_nt(a, b, d)
    try:
        torch.testing.assert_close(c, d, rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        print("v" * 50)
        print(f"assert_close check failed: {e}")
        print("^" * 50)
    acre.cutlass_gemm_nt_naive(a, b, d)
    try:
        torch.testing.assert_close(c, d, rtol=1e-3, atol=1e-2)
    except AssertionError as e:
        print("v" * 50)
        print(f"assert_close check failed: {e}")
        print("^" * 50)
    print("-" * 80)
    return latency


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    # test_gemm(64, 4096, 11008, "fp16")
    test_gemm(4096, 4096, 4096, "fp16")
    test_gemm(8192, 8192, 8192, "fp16")


if __name__ == "__main__":
    main()
