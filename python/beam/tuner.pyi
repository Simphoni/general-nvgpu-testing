from __future__ import annotations
from beam._C.beam_C import cublas_gemmexrc
from beam._C.beam_C import cublas_hgemmrc
from beam._C.beam_C import custom_gemmrc_128x128
from beam._C.beam_C import custom_gemmrc_128x256
from beam._C.beam_C import cutlass_gemmrc
from beam._C.beam_C import cutlass_gemmrc_spec
from beam._C.beam_C import cutlass_gemmrc_splitk
from beam._C.beam_C import cutlass_gemmrc_splitk_spec
from functools import partial
import torch as torch
import triton as triton
__all__ = ['Operator', 'OperatorTuner', 'cublas_gemmexrc', 'cublas_hgemmrc', 'custom_gemmrc_128x128', 'custom_gemmrc_128x256', 'cutlass_gemmrc', 'cutlass_gemmrc_spec', 'cutlass_gemmrc_splitk', 'cutlass_gemmrc_splitk_spec', 'partial', 'torch', 'triton']
class Operator:
    def __init__(self, func: typing.Callable):
        ...
    def __str__(self):
        ...
    def check_callable(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
        ...
    def get_performance(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
        ...
class OperatorTuner:
    def __init__(self):
        ...
    def _get_gemm_specs(self) -> typing.List[beam.tuner.Operator]:
        ...
    def _get_gemm_splitk_specs(self) -> typing.List[beam.tuner.Operator]:
        ...
    def get_hash(self, name: str, args: typing.List) -> str:
        ...
    def tune(self, matmul_shape: typing.List[int]) -> typing.Callable:
        ...
