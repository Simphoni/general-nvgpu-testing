import torch
import time

# from .scripts.gemm_bound import GPU_SPEC_LIST

print(torch.__version__)


NUM_RUNS = 32

STR_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "int8": torch.int8,
    # "int4": torch.int4,
}


def test_gemm(n: int, k: int, m: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    a = torch.randn(n, k, dtype=dtype).cuda()
    b = torch.randn(m, k, dtype=dtype).cuda()
    c = torch.randn(n, m, dtype=dtype).cuda()

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
    print(f"{latency * 1e3} ms")
    return latency


def main():
    test_gemm(4096, 4096, 4096, "fp16")


if __name__ == "__main__":
    main()
