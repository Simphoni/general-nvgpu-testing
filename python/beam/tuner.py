from typing import List, Tuple, Dict, Any, Optional, Callable
from functools import partial

import torch
import triton

from ._C import (
    cublas_gemmexrc,
    cublas_hgemmrc,
    cutlass_gemmrc,
    cutlass_gemmrc_spec,
    cutlass_gemmrc_splitk,
    cutlass_gemmrc_splitk_spec,
    custom_gemmrc_128x128,
    custom_gemmrc_128x256,
)


class Operator:
    def __init__(self, func: Callable):
        self.func = func
    
    def __str__(self):
        return getattr(self.func, "__name__", "Unknown")

    def check_callable(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
        try:
            self.func(a=a, b=b, c=c)
            return True
        except Exception as e:
            return False

    def get_performance(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> float:
        if not self.check_callable(a, b, c):
            return -1
        triton.testing.do_bench(lambda: self.func(a=a, b=b, c=c))
        latency = triton.testing.do_bench(lambda: self.func(a=a, b=b, c=c), warmup=100, rep=100) * 1e-3
        M, N, K = (*c.shape, a.shape[1])
        TFLOPS = 2 * M * N * K / latency * 1e-12
        if TFLOPS > 1000:  # obviously wrong
            return -1
        return TFLOPS


class OperatorTuner:
    gemm_ops: List[Operator]

    cache: Dict[str, Tuple[Callable, float]]

    def _get_gemm_specs(self) -> List[Operator]:
        search_space = [
            ([128, 128], [64, 64]),
            ([128, 256], [64, 64]),
            ([256, 128], [64, 64]),
            ([256, 128], [128, 32]),
            ([256, 128], [32, 128]),
        ]
        ret = []
        for shape_threadblock, shape_warp in search_space:
            ret.append(
                Operator(
                    partial(
                        cutlass_gemmrc_spec,
                        shape_threadblock=shape_threadblock,
                        shape_warp=shape_warp,
                    )
                )
            )
        return ret

    def _get_gemm_splitk_specs(self) -> List[Operator]:
        search_space = [
            ([256, 128], [64, 64]),
            ([128, 128], [64, 64]),
            ([128, 64], [64, 32]),
            ([64, 128], [32, 64]),
            ([64, 64], [32, 32]),
        ]
        ret = []
        for shape_threadblock, shape_warp in search_space:
            for splist_k_slices in range(2, 16):
                ret.append(
                    Operator(
                        partial(
                            cutlass_gemmrc_splitk_spec,
                            shape_threadblock=shape_threadblock,
                            shape_warp=shape_warp,
                            split_k_slices=splist_k_slices,
                        )
                    )
                )
        return ret

    def __init__(self):
        self.cache = {}
        self.gemm_ops = [
            Operator(cublas_gemmexrc),
            # Operator(cublas_hgemmrc),
            Operator(cutlass_gemmrc),
            Operator(cutlass_gemmrc_splitk),
            Operator(custom_gemmrc_128x128),
            Operator(custom_gemmrc_128x256),
        ]
        self.gemm_ops.extend(self._get_gemm_specs())
        self.gemm_ops.extend(self._get_gemm_splitk_specs())

    def get_hash(self, name: str, args: List) -> str:
        ret = name + "("
        for arg in args:
            ret += str(arg) + ","
        ret += ")"
        return ret

    def tune(self, matmul_shape: List[int]) -> Callable:
        hashval = self.get_hash("gemm", matmul_shape)
        if hashval in self.cache:
            return self.cache[hashval]
        best_op = None
        best_tflops = -1
        a = torch.randn(
            matmul_shape[0], matmul_shape[2], dtype=torch.float16, device="cuda"
        )
        b = torch.randn(
            matmul_shape[1], matmul_shape[2], dtype=torch.float16, device="cuda"
        )
        c = torch.randn(
            matmul_shape[0], matmul_shape[1], dtype=torch.float16, device="cuda"
        )
        for op in self.gemm_ops:
            tflops = op.get_performance(a, b, c)
            if tflops > best_tflops:
                best_op = op
                best_tflops = tflops
        self.cache[hashval] = (best_op, best_tflops)
        return self.cache[hashval]
