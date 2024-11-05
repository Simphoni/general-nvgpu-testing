import torch
import time
import colorama

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


import acre.perf as acre

colorama.init(autoreset=True)

def run_and_check(func, a, b, c, d):
    torch.randn(d.size(), out=d)
    func(a, b, d)
    try:
        torch.testing.assert_close(c, d, rtol=2e-2, atol=2e-2)
    except AssertionError as e:
        print("v" * 40)
        print(colorama.Fore.RED + f"{func.__name__} failure: {e}")
        print(c)
        print(d)
        print((d * d).max())
        print("^" * 40)

def test_gemm(m: int, n: int, k: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    a = torch.randn(m, k, dtype=dtype).cuda()
    b = torch.randn(n, k, dtype=dtype).cuda()
    c = torch.randn(m, n, dtype=dtype).cuda()

    torch.autograd.set_grad_enabled(False)
    print(f"[TEST] GEMM: {m=}, {n=}, {k=}, {dtype=}")

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
    print(f"PyTorch: {m * n * k * 2 / latency / 1e12} TFLOPS")

    d = torch.zeros_like(c)
    # run_and_check(acre.cublas_gemm_nt, a, b, c, d)
    # run_and_check(acre.cublas_gemmex_nt, a, b, c, d)
    # run_and_check(acre.cutlass_gemm_nt_naive, a, b, c, d)
    # run_and_check(acre.cutlass_gemm_nt_manual_tune, a, b, c, d)
    run_and_check(acre.cutlass_parallel_gemmrc_layernorm, a, b, c, d)
     
    print("-" * 80)
    
    return latency


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    # test_gemm(64, 4096, 11008, "fp16")
    test_gemm(4096, 4096, 4096, "fp16")
    #test_gemm(8192, 4096, 8192, "fp16")


if __name__ == "__main__":
    main()
