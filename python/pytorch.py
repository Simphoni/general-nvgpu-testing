import torch
import time
import colorama
import argparse
from functools import partial
import triton

import acre

print(torch.__version__)


NUM_RUNS = 64

SLEEP_MILLISEC_BEFORE_EVAL = 100

WITH_NCU = False

if WITH_NCU:
    NUM_RUNS = 2

STR_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    # "int4": torch.int4,
}


colorama.init(autoreset=True)


def check(func, answers, outputs):
    assert len(outputs) == len(answers), f"{len(outputs)=} != {len(answers)=}"
    length = len(outputs)
    for i in range(length):
        assert (
            outputs[i].dtype == answers[i].dtype
        ), f"{i=}, {outputs[i].dtype=}, {answers[i].dtype=}"
        assert (
            outputs[i].shape == answers[i].shape
        ), f"{i=}, {outputs[i].shape=}, {answers[i].shape=}"

    if WITH_NCU:
        return

    errors = []
    for i in range(length):
        c = answers[i]
        d = outputs[i]
        try:
            torch.testing.assert_close(actual=d, expected=c, rtol=2e-2, atol=2e-2)
        except AssertionError as err:
            errors.append((i, err))
    if len(errors) == 0:
        print(colorama.Fore.GREEN + f"{func.__name__} success")
        return
    for itensor, err in errors:
        se = str(err)
        percent = 100
        for i in range(0, len(se)):
            if se[i] == "%":
                end = i
                begin = end - 1
                while se[begin] != "(":
                    begin -= 1
                percent = float(se[begin + 1 : end])
        if percent < 0.12:
            print(
                colorama.Fore.YELLOW
                + f"{func.__name__} tensor({itensor}): mismatch percent = {percent}%"
            )
        else:
            print(f"{answers[itensor]=}")
            print(f"{outputs[itensor]=}")
            print(colorama.Fore.RED + f"{func.__name__} tensor({itensor}): {err}")


def run_perf(func, inputs, outputs):
    return triton.testing.do_bench(lambda: func(*(inputs + outputs)), warmup=50, rep=100) * 1e-3
    # for _ in range(NUM_RUNS):
    #     func(*(inputs + outputs))
    # torch.cuda.synchronize()
    # time.sleep(SLEEP_MILLISEC_BEFORE_EVAL * 1e-3)
    # tic = time.time()
    # for _ in range(NUM_RUNS):
    #     func(*(inputs + outputs))
    # torch.cuda.synchronize()
    # toc = time.time()
    # latency = (toc - tic) / NUM_RUNS
    # return latency


def universal_test(
    ansfunc, torchfunc, cutefuncs, input_shapes, output_shapes, dtype, metric
):
    inputs = [torch.randn(shape, dtype=dtype, device="cuda") for shape in input_shapes]
    outputs = [
        torch.zeros(shape, dtype=dtype, device="cuda") for shape in output_shapes
    ]
    answers = [
        torch.zeros(shape, dtype=dtype, device="cuda") for shape in output_shapes
    ]

    ansfunc(*(inputs + answers))

    latency = run_perf(torchfunc, inputs, outputs)
    print(f"PyTorch: {latency * 1e3:.6f} ms, [{metric(latency=latency):.3f}] TFLOPS")

    for func in cutefuncs:
        for d in outputs:
            d.zero_()
        latency = run_perf(func, inputs, outputs)
        print(
            f"{func.__name__}: {latency * 1e3:.6f} ms, [{metric(latency=latency):.3f}] TFLOPS"
        )
        check(func, answers, outputs)


def test_gemm(m: int, n: int, k: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    print(f"[TEST] GEMM: {m=}, {n=}, {k=}, {dtype=}")

    def torchfunc(a, b, c):
        torch.matmul(a, b.t(), out=c)

    def compute_flops(m, n, k, latency):
        return 2 * m * n * k / latency * 1e-12

    universal_test(
        torchfunc,
        torchfunc,
        [
            acre.cutlass_parallel_gemmrc,
            # acre.cublas_hgemmrc,
            # acre.cublas_gemmexrc,
            # acre.cutlass_gemmrc_naive,
            # acre.cutlass_gemmrc_spec,
            acre.cutlass_gemmrc_splitk,
        ],
        [(m, k), (n, k)],
        [(m, n)],
        dtype,
        partial(compute_flops, m, n, k),
    )
    print("-" * 80)


def main():
    test_gemm(4096, 4096, 4096, "fp16")
    # test_gemm(4096, 4096, 4096, "fp16")

    test_gemm(512, 512, 8192, "fp16")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_grad_enabled(False)
    print(f"Device: {torch.cuda.get_device_name()}")
    # acre.set_default_nrep(NUM_RUNS)
    main()
