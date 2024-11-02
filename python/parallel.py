import torch
import time
import colorama


import acre.perf as acre

A = torch.randn(512, 4096, dtype=torch.float16, device="cuda")

acre.cutlass_parallel_gemmcr_layernorm(A, A, A)