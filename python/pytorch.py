import torch
import time
import colorama
import argparse

import acre.perf as acre

print(torch.__version__)


NUM_RUNS = 64

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

argparse.ArgumentParser

colorama.init(autoreset=True)


def run_and_check(func, inputs, answers, outputs):
    assert len(outputs) == len(answers), f"{len(outputs)=} != {len(answers)=}"
    length = len(outputs)
    for i in range(length):
        assert (
            outputs[i].dtype == answers[i].dtype
        ), f"{i=}, {outputs[i].dtype=}, {answers[i].dtype=}"
        assert (
            outputs[i].shape == answers[i].shape
        ), f"{i=}, {outputs[i].shape=}, {answers[i].shape=}"
    for d in outputs:
        torch.zeros(d.size(), out=d)
    func(*(inputs + outputs))

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
    print("v" * 40)
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
        if percent < 0.1:
            print(
                colorama.Fore.YELLOW
                + f"{func.__name__} tensor({itensor}): mismatch percent = {percent}%"
            )
        else:
            print(answers[itensor])
            print(outputs[itensor])
            print(colorama.Fore.RED + f"{func.__name__} tensor({itensor}): {err}")
    print("^" * 40)


def universal_test(ansfunc, torchfunc, cutefuncs, input_shapes, output_shapes, dtype):
    inputs = [torch.randn(shape, dtype=dtype, device="cuda") for shape in input_shapes]
    outputs = [
        torch.zeros(shape, dtype=dtype, device="cuda") for shape in output_shapes
    ]
    answers = [
        torch.zeros(shape, dtype=dtype, device="cuda") for shape in output_shapes
    ]

    for _ in range(NUM_RUNS):
        torchfunc(*(inputs + answers))
    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(NUM_RUNS):
        torchfunc(*(inputs + answers))
    torch.cuda.synchronize()
    toc = time.time()
    latency = (toc - tic) / NUM_RUNS
    print(f"PyTorch: {latency * 1e3} ms")

    ansfunc(*(inputs + answers))
    for func in cutefuncs:
        run_and_check(func, inputs, answers, outputs)


def test_gemm(m: int, n: int, k: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    print(f"[TEST] GEMM: {m=}, {n=}, {k=}, {dtype=}")

    def torchfunc(a, b, c):
        torch.matmul(a, b.t(), out=c)

    universal_test(
        torchfunc,
        torchfunc,
        [
            # acre.cublas_gemm_nt,
            # acre.cublas_gemmex_nt,
            # acre.cutlass_gemm_nt_naive,
            # acre.cutlass_gemm_nt_manual_tune,
            acre.cutlass_parallel_gemmrc,
        ],
        [(m, k), (n, k)],
        [(m, n)],
        dtype,
    )
    print("-" * 80)


def test_gemm_ln(gemmM, gemmN, gemmK, lnM, lnN, dtype):
    dtype = STR_DTYPE_MAPPING[dtype]
    print(f"[TEST] GEMM_LN: {gemmM=}, {gemmN=}, {gemmK=}, {lnM=}, {lnN=}, {dtype=}")

    def ansfunc(a, b, c, d, e):
        torch.matmul(a, b.t(), out=d)
        e.copy_(torch.nn.functional.relu(c))

    def torchfunc(a, b, c, d, e):
        torch.matmul(a, b.t(), out=d)
        torch.nn.functional.silu(c)

    def acrefunc(a, b, c, d, e):
        # acre: (a,b)->d, (c)->e
        acre.cutlass_parallel_gemmrc_lnr(a, b, d, c, e)

    universal_test(
        ansfunc,
        torchfunc,
        [acrefunc],
        [(gemmM, gemmK), (gemmN, gemmK), (lnM, lnN)],
        [(gemmM, gemmN), (lnM, lnN)],
        dtype,
    )
    print("-" * 80)


def main():
    test_gemm_ln(4096, 4096, 4096, 4096 * 2, 4096, "fp16")
    test_gemm(4096, 4096, 4096, "fp16")
    # test_gemm(8192, 4096, 8192, "fp16")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_grad_enabled(False)
    print(f"Device: {torch.cuda.get_device_name()}")
    acre.set_default_nrep(NUM_RUNS)
    main()
