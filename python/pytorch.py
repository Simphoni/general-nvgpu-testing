import torch
import time
import colorama
import argparse
from functools import partial
import triton

import matplotlib.pyplot as plt
import numpy as np

import beam

plt.rcParams["font.family"] = "DejaVu Sans"

print(torch.__version__)


SLEEP_MILLISEC_BEFORE_EVAL = 100

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
    name = getattr(func, "__name__", "unknown")
    assert len(outputs) == len(answers), f"{len(outputs)=} != {len(answers)=}"
    length = len(outputs)
    for i in range(length):
        assert (
            outputs[i].dtype == answers[i].dtype
        ), f"{i=}, {outputs[i].dtype=}, {answers[i].dtype=}"
        assert (
            outputs[i].shape == answers[i].shape
        ), f"{i=}, {outputs[i].shape=}, {answers[i].shape=}"

    errors = []
    for i in range(length):
        c = answers[i]
        d = outputs[i]
        try:
            torch.testing.assert_close(actual=d, expected=c, rtol=2e-2, atol=2e-2)
        except AssertionError as err:
            errors.append((i, err))
    if len(errors) == 0:
        print(colorama.Fore.GREEN + f"{name} success")
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
                + f"{name} tensor({itensor}): mismatch percent = {percent}%"
            )
        else:
            print(f"{answers[itensor]=}")
            print(f"{outputs[itensor]=}")
            print(colorama.Fore.RED + f"{name} tensor({itensor}): {err}")


def run_perf(func, inputs, outputs):
    time.sleep(0.05)
    return triton.testing.do_bench(lambda: func(*(inputs + outputs))) * 1e-3


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

    retdict = {}
    retdict["torch"] = metric(latency=latency)
    retdict["custom"] = []

    for func in cutefuncs:
        for d in outputs:
            d.zero_()
        latency = run_perf(func, inputs, outputs)
        name = getattr(func, "__name__", "unknown")
        print(f"{name}: {latency * 1e3:.6f} ms, [{metric(latency=latency):.3f}] TFLOPS")
        check(func, answers, outputs)
        retdict["custom"].append(metric(latency=latency))

    return retdict


def test_gemm_splitk(m: int, n: int, k: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    print(f"[TEST] GEMM SPLITK: {m=}, {n=}, {k=}, {dtype=}")

    def torchfunc(a, b, c):
        torch.matmul(a, b.t(), out=c)

    def compute_flops(m, n, k, latency):
        return 2 * m * n * k / latency * 1e-12

    funclist = []
    for i in range(1, 16):
        # funclist.append(
        #     partial(
        #         beam.cutlass_gemmrc_splitk_spec,
        #         shape_threadblock=[64, 64],
        #         split_k_slices=i,
        #     )
        # )
        funclist.append(
            partial(
                beam.cutlass_gemmrc_splitk_spec,
                shape_threadblock=[128, 64],
                shape_warp=[64, 32],
                split_k_slices=i,
            )
        )

    rets = universal_test(
        torchfunc,
        torchfunc,
        funclist,
        [(m, k), (n, k)],
        [(m, n)],
        dtype,
        partial(compute_flops, m, n, k),
    )
    print("-" * 80)

    plt.figure(figsize=(10, 6))
    plt.plot(rets["custom"], marker="o", label="custom")
    plt.plot([rets["torch"]] * len(rets["custom"]), label="torch")
    plt.xlabel("split_k_slices")
    plt.ylabel("TFLOPS")
    plt.title("split_k_slices vs TFLOPS")
    plt.legend()
    plt.savefig("splitk.png")
    plt.close()


def test_gemm(m: int, n: int, k: int, dtype: str):
    dtype = STR_DTYPE_MAPPING[dtype]
    print(f"[TEST] GEMM: {(m,n,k)=}, {dtype=}")

    def torchfunc(a, b, c):
        torch.matmul(a, b.t(), out=c)

    def compute_flops(m, n, k, latency):
        return 2 * m * n * k / latency * 1e-12

    def spec(a, b, c):
        beam.cutlass_gemmrc_spec(
            a, b, c, shape_threadblock=[256, 128], shape_warp=[128, 32]
        )

    universal_test(
        torchfunc,
        torchfunc,
        [
            # beam.cublas_hgemmrc,
            # beam.cublas_gemmexrc,
            beam.custom_gemmrc_128x256,
            beam.cutlass_gemmrc,
            # spec,
            # beam.custom_gemmrc_128x128,
            # beam.cutlass_gemmrc_spec,
        ],
        [(m, k), (n, k)],
        [(m, n)],
        dtype,
        partial(compute_flops, m, n, k),
    )
    print("-" * 80)


def main():
    # test_gemm(4096, 4096, 4096, "fp16")
    test_gemm(4096, 4096, 4096, "fp16")

    # test_gemm_splitk(512, 512, 8192, "fp16")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_grad_enabled(False)
    print(f"Device: {torch.cuda.get_device_name()}")
    main()
