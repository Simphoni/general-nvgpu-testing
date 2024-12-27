import triton
import triton.language as tl
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Kernel for matrix multiplication."""
    # -----------------------------------------------------------
    # Matrix multiplication kernel
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Initialize the accumulator to zero
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    for k in range(0, K, BLOCK_K):
        # Load the next block of A and B, using masking to handle bounds
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        # Perform the matrix multiplication on the loaded blocks
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # -----------------------------------------------------------
    # Write back the result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor):
    """Matrix multiplication using Triton kernel."""
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def benchmark_matmul(M, N, K, provider):
    """Benchmark matmul for specific sizes."""
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: matmul(a, b), warmup=50, rep=100)
    elif provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=50, rep=100)
    
    return M * N * K * 2 / 1e12 / (ms * 1e-3)

def main():
    """Main function to run benchmarks and create visualizations."""
    sizes = [128, 256, 512, 1024, 2048, 4096]
    providers = ['triton', 'torch']
    
    # Test different matrix shapes
    for shape_type in ['square']:
        perf_results = {provider: [] for provider in providers}
        
        for size in sizes:
            if shape_type == 'square':
                M, N, K = size, size, size
            elif shape_type == 'tall':
                M, N, K = size * 2, size, size
            else:  # wide
                M, N, K = size, size * 2, size
            
            for provider in providers:
                perf = benchmark_matmul(M, N, K, provider)
                perf_results[provider].append(perf)
        
        print(perf_results)
        # Create performance plot
        plt.figure(figsize=(10, 6))
        for provider in providers:
            plt.plot(sizes, perf_results[provider], marker='o', label=provider)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (ms)')
        plt.title(f'GEMM Performance ({shape_type} matrices)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'perf-{shape_type}.png')
        plt.close()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
