#!/usr/bin/env python3
"""
Triton / PyTorch implementation of the 3D tensor product  C = Ar * (As * (At * B))
used in spectral element methods.

Usage:  python triton_tensor_product.py M K [--N 65536] [--ntrys 10]
"""

import argparse
import time
import numpy as np

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ─────────────────────────────────────────────────────────────────────
# CPU reference (numpy einsum, 3‑stage sum‑factorised)
# ─────────────────────────────────────────────────────────────────────

def cpu_tensor_product(Ar, As, At, B):
    """
    Ar, As, At : (M, K)   row‑major
    B          : (N, K, K, K)
    returns C  : (N, M, M, M)
    """
    tmp1 = np.einsum('mk,nijk->nijm', At, B)      # contract last  K → M
    tmp2 = np.einsum('mj,nijc->nimc', As, tmp1)    # contract middle K → M
    C    = np.einsum('mi,nibc->nmbc', Ar, tmp2)    # contract first K → M
    return C


# ─────────────────────────────────────────────────────────────────────
# PyTorch GPU reference (torch.einsum)
# ─────────────────────────────────────────────────────────────────────

def torch_tensor_product(Ar, As, At, B):
    tmp1 = torch.einsum('mk,nijk->nijm', At, B)
    tmp2 = torch.einsum('mj,nijc->nimc', As, tmp1)
    C    = torch.einsum('mi,nibc->nmbc', Ar, tmp2)
    return C


# ─────────────────────────────────────────────────────────────────────
# Triton kernels  (three separate stages)
# ─────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _stage1_kernel(
        At_ptr, B_ptr, out_ptr,
        N: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out[n,i,j,m] = sum_k At[m,k] * B[n,i,j,k]"""
        n = tl.program_id(0)
        offs = tl.arange(0, BLOCK)          # lanes cover i*j*m
        total = K * K * M
        mask = offs < total
        # decompose lane → (i, j, m)
        m = offs % M
        tmp = offs // M
        j = tmp % K
        i = tmp // K
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for kk in tl.static_range(K):
            a_val = tl.load(At_ptr + m * K + kk, mask=mask)          # At[m, kk]
            b_val = tl.load(B_ptr + n * K*K*K + i * K*K + j * K + kk, mask=mask)  # B[n,i,j,kk]
            acc += a_val * b_val
        tl.store(out_ptr + n * total + offs, acc, mask=mask)

    @triton.jit
    def _stage2_kernel(
        As_ptr, in_ptr, out_ptr,
        N: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out[n,i,b,c] = sum_j As[b,j] * in[n,i,j,c]"""
        n = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        total = K * M * M
        mask = offs < total
        c = offs % M
        tmp = offs // M
        b = tmp % M
        i = tmp // M
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for jj in tl.static_range(K):
            a_val = tl.load(As_ptr + b * K + jj, mask=mask)
            in_val = tl.load(in_ptr + n * K*K*M + i * K*M + jj * M + c, mask=mask)
            acc += a_val * in_val
        tl.store(out_ptr + n * total + offs, acc, mask=mask)

    @triton.jit
    def _stage3_kernel(
        Ar_ptr, in_ptr, out_ptr,
        N: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out[n,a,b,c] = sum_i Ar[a,i] * in[n,i,b,c]"""
        n = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        total = M * M * M
        mask = offs < total
        c = offs % M
        tmp = offs // M
        b = tmp % M
        a = tmp // M
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for ii in tl.static_range(K):
            a_val = tl.load(Ar_ptr + a * K + ii, mask=mask)
            in_val = tl.load(in_ptr + n * K*M*M + ii * M*M + b * M + c, mask=mask)
            acc += a_val * in_val
        tl.store(out_ptr + n * total + offs, acc, mask=mask)


def triton_tensor_product(Ar, As, At, B, M, K, N):
    """Run the three Triton stage kernels.  All tensors must be on CUDA."""
    BLOCK1 = next_power_of_2(K * K * M)
    BLOCK2 = next_power_of_2(K * M * M)
    BLOCK3 = next_power_of_2(M * M * M)

    tmp1 = torch.empty((N, K, K, M), dtype=torch.float64, device='cuda')
    tmp2 = torch.empty((N, K, M, M), dtype=torch.float64, device='cuda')
    C    = torch.empty((N, M, M, M), dtype=torch.float64, device='cuda')

    grid = (N,)
    _stage1_kernel[grid](At, B, tmp1, N, M, K, BLOCK1)
    _stage2_kernel[grid](As, tmp1, tmp2, N, M, K, BLOCK2)
    _stage3_kernel[grid](Ar, tmp2, C, N, M, K, BLOCK3)
    return C


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Triton 3D tensor product benchmark')
    parser.add_argument('M', type=int, help='output polynomial order + 1')
    parser.add_argument('K', type=int, help='input polynomial order + 1')
    parser.add_argument('--N', type=int, default=65536, help='number of elements (default 65536)')
    parser.add_argument('--ntrys', type=int, default=10, help='timing iterations')
    args = parser.parse_args()

    M, K, N, ntrys = args.M, args.K, args.N, args.ntrys

    print(f"M={M}, K={K}, N={N}, ntrys={ntrys}")
    FLOP = N * (K*K*M + K*M*M + M*M*M) * 2 * K

    # ── data generation ──────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    Ar = rng.uniform(-1, 1, (M, K))
    As = rng.uniform(-1, 1, (M, K))
    At = rng.uniform(-1, 1, (M, K))
    B  = rng.uniform(-1, 1, (N, K, K, K))

    # ── CPU reference ────────────────────────────────────────────────
    print("\n--- CPU numpy.einsum reference ---")
    C_cpu = cpu_tensor_product(Ar, As, At, B)
    print(f"  C shape: {C_cpu.shape}")

    # ── GPU setup ────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU benchmarks.")
        return

    dev = torch.cuda.get_device_properties(0)
    mem = getattr(dev, 'total_mem', None) or getattr(dev, 'total_memory', 0)
    print(f"\nGPU: {dev.name}  (SM {dev.major}.{dev.minor}, {mem // 2**20} MB)")

    Ar_t = torch.from_numpy(Ar).cuda()
    As_t = torch.from_numpy(As).cuda()
    At_t = torch.from_numpy(At).cuda()
    B_t  = torch.from_numpy(B).cuda()

    # ── torch.einsum on GPU ──────────────────────────────────────────
    print("\n--- torch.einsum GPU ---")
    C_torch = torch_tensor_product(Ar_t, As_t, At_t, B_t)
    torch.cuda.synchronize()

    diff_torch = np.max(np.abs(C_cpu - C_torch.cpu().numpy()))
    tol = 1e-12
    status = "PASS" if diff_torch <= tol else "FAIL"
    print(f"  CPU vs torch.einsum  max|diff| = {diff_torch:.3e}  [{status}]")

    # warm-up + timing
    for _ in range(3):
        torch_tensor_product(Ar_t, As_t, At_t, B_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ntrys):
        torch_tensor_product(Ar_t, As_t, At_t, B_t)
    torch.cuda.synchronize()
    dt_torch = (time.perf_counter() - t0) / ntrys
    print(f"  time = {dt_torch:.4e} s   TFLOPS = {FLOP / dt_torch / 1e12:.3f}")

    # ── Triton kernels ───────────────────────────────────────────────
    if not HAS_TRITON:
        print("\nTriton not installed — skipping Triton benchmark.")
        return

    print("\n--- Triton kernels ---")
    try:
        C_tri = triton_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()

        diff_tri = np.max(np.abs(C_cpu - C_tri.cpu().numpy()))
        status = "PASS" if diff_tri <= tol else "FAIL"
        print(f"  CPU vs Triton        max|diff| = {diff_tri:.3e}  [{status}]")

        # warm-up + timing
        for _ in range(3):
            triton_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ntrys):
            triton_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()
        dt_tri = (time.perf_counter() - t0) / ntrys
        print(f"  time = {dt_tri:.4e} s   TFLOPS = {FLOP / dt_tri / 1e12:.3f}")

    except Exception as e:
        print(f"  Triton compilation failed (likely SM < 7.0): {e}")
        print("  Falling back to torch.einsum results above.")


if __name__ == '__main__':
    main()
