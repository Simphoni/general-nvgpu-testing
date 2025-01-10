from dataclasses import dataclass


@dataclass
class GPUSpec:
    name: str
    bw: float  # GB/s
    fp32_rt: float  # RT core
    # Tensor core
    tf32: float
    fp16: float
    int8: float
    fp8: float = 0


BITWIDTH_MAP = {
    "tf32": 32,
    "fp16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}

DTYPE_FALLBACK_ORDER = {"fp16", "fp32"}

GPU_SPEC_LIST = [
    GPUSpec("A100_{PCIe,SXM}_HBM2", 1555, 19.5, 156, 312, 624),
    GPUSpec("A100_PCIe_HBM2e", 1935, 19.5, 156, 312, 624),
    GPUSpec("A100_SXM_HBM2e", 2039, 19.5, 156, 312, 624),
    GPUSpec("H100_SXM", 3350, 67, 989 / 2, 1979 / 2, 3958 / 2, 3958 / 2),
    GPUSpec("H100_NVL", 3900, 60, 835 / 2, 1671 / 2, 3341 / 2, 3341 / 2),
]

GPU_NAME_SPEC_MAP = {spec.name: spec for spec in GPU_SPEC_LIST}


def print_gemm_bound(
    spec: GPUSpec,
    dtype: str,
    n: int,
    k: int,
    m: int,
    micro_n: int,
    micro_k: int,
    micro_m: int,
):
    assert dtype in BITWIDTH_MAP
    bw = spec.bw * 1e9  # GB/s -> B/s
    dtype_size = BITWIDTH_MAP[dtype] / 8
    rt = getattr(spec, dtype) * 1e12

    flops = n * k * m * 2
    compute_bound = flops / rt * 1000
    jobs = n / micro_n * k / micro_k * m / micro_m
    memory_bound = (
        jobs
        * (
            micro_n * micro_k + micro_k * micro_m + micro_n * micro_m * 2
        )  # 2 for atomicAdd
        * dtype_size
        / bw
        * 1000
    )

    print(spec.name)
    print(f"\t{dtype} n={n} k={k} m={m}")
    print(f"\tmicro_n={micro_n} micro_k={micro_k} micro_m={micro_m}")
    print(f"\tparallel degree={jobs}")
    print(f"\tCompute bound: {compute_bound:.2f} ms")
    print(f"\tMemory bound: {memory_bound:.2f} ms")
    print(f"\tCompute to Memory ratio: {compute_bound / memory_bound:.2f}")


# A100 on lotus1
spec = GPU_NAME_SPEC_MAP["A100_{PCIe,SXM}_HBM2"]
print_gemm_bound(spec, "fp16", 4096, 4096, 4096, 512, 512, 1024)

# for spec in GPU_SPEC_LIST:
#     for dtype in ["fp16", "int8"]:
#         print_gemm_bound(spec, dtype, 4096, 4096, 4096, 256, 256, 128)
