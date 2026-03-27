#!/usr/bin/env python3
"""
Triton / PyTorch implementation of the 3D tensor product  C = Ar * (As * (At * B))
used in spectral element methods.

Usage:  python triton_tensor_product.py M K [--N 65536] [--ntrys 10]
"""

import argparse
import glob
import os
import time
import numpy as np

# Ensure venv-installed nvidia tools (ptxas) are found before system ones
for _sp in __import__('site').getsitepackages():
    for _p in glob.glob(os.path.join(_sp, 'nvidia', '*', 'bin')):
        if _p not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _p + ':' + os.environ.get('PATH', '')

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# try:
#     import warp as wp
#     wp.init()
#     HAS_WARP = True
# except (ImportError, Exception):
#     HAS_WARP = False
HAS_WARP = False

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


def bench(fn, ntrys, *args, **kwargs):
    """Warm up (3 iters), then time ntrys iterations.  Returns seconds/iter."""
    for _ in range(3):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ntrys):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ntrys


def validate(label, C_cpu, C_gpu, tol=1e-12):
    diff = np.max(np.abs(C_cpu - C_gpu.cpu().numpy()))
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  CPU vs {label:20s} max|diff| = {diff:.3e}  [{status}]")
    return diff <= tol


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
# torch.compile'd einsum
# ─────────────────────────────────────────────────────────────────────

torch_tensor_product_compiled = torch.compile(torch_tensor_product)


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
# Triton fused kernel — single launch, L1-cached intermediates
# ─────────────────────────────────────────────────────────────────────
# One program per batch element.  Three stages in a single kernel;
# intermediates (tmpK, tmpJ) go through a per-element scratch buffer
# that stays warm in L1 cache (~3 KB per element).

if HAS_TRITON:

    @triton.jit
    def _fused_tp_kernel(
        Ar_ptr, As_ptr, At_ptr, B_ptr, C_ptr, scratch_ptr,
        N: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        offs = tl.arange(0, BLOCK)

        KKM = K * K * M
        KMM = K * M * M
        MMM = M * M * M
        K3  = K * K * K

        # Per-element scratch: tmpK[0..KKM-1] then tmpJ[KKM..KKM+KMM-1]
        tmpK_ptr = scratch_ptr + n * (KKM + KMM)
        tmpJ_ptr = tmpK_ptr + KKM

        # ── Stage 1: tmpK[i,j,m] = sum_k At[m,k] * B[i,j,k] ────────
        mask1 = offs < KKM
        m1 = offs % M
        tmp1 = offs // M
        j1 = tmp1 % K
        i1 = tmp1 // K
        acc1 = tl.zeros([BLOCK], dtype=tl.float64)
        for kk in tl.static_range(K):
            at_val = tl.load(At_ptr + m1 * K + kk, mask=mask1)
            b_val  = tl.load(B_ptr + n * K3 + i1 * K * K + j1 * K + kk,
                             mask=mask1)
            acc1 += at_val * b_val
        tl.store(tmpK_ptr + offs, acc1, mask=mask1)
        tl.debug_barrier()

        # ── Stage 2: tmpJ[i,b,c] = sum_j As[b,j] * tmpK[i,j,c] ─────
        S2_PASSES: tl.constexpr = (K * M * M + BLOCK - 1) // BLOCK
        for t in tl.static_range(S2_PASSES):
            idx2 = offs + t * BLOCK
            mask2 = idx2 < KMM
            c2 = idx2 % M
            tmp2 = idx2 // M
            b2 = tmp2 % M
            i2 = tmp2 // M
            acc2 = tl.zeros([BLOCK], dtype=tl.float64)
            for jj in tl.static_range(K):
                as_val   = tl.load(As_ptr  + b2 * K + jj, mask=mask2)
                tmpk_val = tl.load(tmpK_ptr + i2 * K * M + jj * M + c2,
                                   mask=mask2)
                acc2 += as_val * tmpk_val
            tl.store(tmpJ_ptr + idx2, acc2, mask=mask2)
        tl.debug_barrier()

        # ── Stage 3: C[n,a,b,c] = sum_i Ar[a,i] * tmpJ[i,b,c] ──────
        S3_PASSES: tl.constexpr = (M * M * M + BLOCK - 1) // BLOCK
        for t in tl.static_range(S3_PASSES):
            idx3 = offs + t * BLOCK
            mask3 = idx3 < MMM
            c3 = idx3 % M
            tmp3 = idx3 // M
            b3 = tmp3 % M
            a3 = tmp3 // M
            acc3 = tl.zeros([BLOCK], dtype=tl.float64)
            for ii in tl.static_range(K):
                ar_val   = tl.load(Ar_ptr  + a3 * K + ii, mask=mask3)
                tmpj_val = tl.load(tmpJ_ptr + ii * M * M + b3 * M + c3,
                                   mask=mask3)
                acc3 += ar_val * tmpj_val
            tl.store(C_ptr + n * MMM + idx3, acc3, mask=mask3)


def triton_fused_tensor_product(Ar, As, At, B, M, K, N,
                                num_warps=4, BLOCK=None):
    """Run the fused Triton kernel.  All tensors must be on CUDA."""
    KKM = K * K * M
    KMM = K * M * M
    MMM = M * M * M
    if BLOCK is None:
        BLOCK = next_power_of_2(max(KKM, KMM, MMM))

    scratch = torch.empty(N * (KKM + KMM), dtype=torch.float64, device='cuda')
    C = torch.empty((N, M, M, M), dtype=torch.float64, device='cuda')

    _fused_tp_kernel[(N,)](Ar, As, At, B, C, scratch, N, M, K, BLOCK,
                           num_warps=num_warps)
    return C


# ─────────────────────────────────────────────────────────────────────
# Triton batched fused kernel: multiple elements per program
# ─────────────────────────────────────────────────────────────────────
# Same 3-stage sum-factorisation as _fused_tp_kernel but each program
# processes BATCH elements sequentially, reusing the same scratch space.
# Reduces program count → less scheduling overhead.

if HAS_TRITON:

    @triton.jit
    def _fused_batched_kernel(
        Ar_ptr, As_ptr, At_ptr, B_ptr, C_ptr, scratch_ptr,
        N: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr, BATCH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)

        KKM: tl.constexpr = K * K * M
        KMM: tl.constexpr = K * M * M
        MMM: tl.constexpr = M * M * M
        K3: tl.constexpr = K * K * K

        # Scratch is per-program (reused across batch elements)
        tmpK_ptr = scratch_ptr + pid * (KKM + KMM)
        tmpJ_ptr = tmpK_ptr + KKM

        S2P: tl.constexpr = (KMM + BLOCK - 1) // BLOCK
        S3P: tl.constexpr = (MMM + BLOCK - 1) // BLOCK

        # Pre-compute lane mappings (constant across batch iterations)
        mask1 = offs < KKM
        m1 = offs % M
        tmp1 = offs // M
        j1 = tmp1 % K
        i1 = tmp1 // K

        for elem in tl.static_range(BATCH):
            n = pid * BATCH + elem

            # ── Stage 1: tmpK[i,j,m] = sum_k At[m,k] * B[n,i,j,k] ──
            acc1 = tl.zeros([BLOCK], dtype=tl.float64)
            for kk in tl.static_range(K):
                at_val = tl.load(At_ptr + m1 * K + kk, mask=mask1)
                b_val  = tl.load(B_ptr + n * K3
                                 + i1 * K * K + j1 * K + kk, mask=mask1)
                acc1 += at_val * b_val
            tl.store(tmpK_ptr + offs, acc1, mask=mask1)
            tl.debug_barrier()

            # ── Stage 2: tmpJ[i,b,c] = sum_j As[b,j] * tmpK[i,j,c] ─
            for t in tl.static_range(S2P):
                idx2 = offs + t * BLOCK
                mask2 = idx2 < KMM
                c2 = idx2 % M
                tmp2 = idx2 // M
                b2 = tmp2 % M
                i2 = tmp2 // M
                acc2 = tl.zeros([BLOCK], dtype=tl.float64)
                for jj in tl.static_range(K):
                    as_val   = tl.load(As_ptr  + b2 * K + jj, mask=mask2)
                    tmpk_val = tl.load(tmpK_ptr + i2 * K * M + jj * M + c2,
                                       mask=mask2)
                    acc2 += as_val * tmpk_val
                tl.store(tmpJ_ptr + idx2, acc2, mask=mask2)
            tl.debug_barrier()

            # ── Stage 3: C[n,a,b,c] = sum_i Ar[a,i] * tmpJ[i,b,c] ──
            for t in tl.static_range(S3P):
                idx3 = offs + t * BLOCK
                mask3 = idx3 < MMM
                c3 = idx3 % M
                tmp3 = idx3 // M
                b3 = tmp3 % M
                a3 = tmp3 // M
                acc3 = tl.zeros([BLOCK], dtype=tl.float64)
                for ii in tl.static_range(K):
                    ar_val   = tl.load(Ar_ptr  + a3 * K + ii, mask=mask3)
                    tmpj_val = tl.load(tmpJ_ptr + ii * M * M + b3 * M + c3,
                                       mask=mask3)
                    acc3 += ar_val * tmpj_val
                tl.store(C_ptr + n * MMM + idx3, acc3, mask=mask3)
            # No barrier needed after Stage 3: next iteration's Stage 1
            # only writes tmpK (doesn't read tmpJ), and the Stage 2
            # barrier already ensured all tmpK reads completed.


def triton_batched_tensor_product(Ar, As, At, B, M, K, N,
                                  BATCH=8, num_warps=4, BLOCK=None):
    """Batched fused Triton kernel: BATCH elements per program."""
    KKM = K * K * M
    KMM = K * M * M
    MMM = M * M * M
    if BLOCK is None:
        BLOCK = next_power_of_2(max(KKM, KMM, MMM))

    n_progs = N // BATCH
    scratch = torch.empty(n_progs * (KKM + KMM), dtype=torch.float64,
                          device='cuda')
    C = torch.empty((N, M, M, M), dtype=torch.float64, device='cuda')

    _fused_batched_kernel[(n_progs,)](
        Ar, As, At, B, C, scratch, N, M, K, BLOCK, BATCH,
        num_warps=num_warps)
    return C


# ─────────────────────────────────────────────────────────────────────
# NVIDIA Warp kernels (commented out for multi-size benchmarking)
# ─────────────────────────────────────────────────────────────────────

# if HAS_WARP:
#
#     @wp.kernel
#     def _warp_stage1( ... ): ...
#     @wp.kernel
#     def _warp_stage2( ... ): ...
#     @wp.kernel
#     def _warp_stage3( ... ): ...
#
# def warp_tensor_product( ... ): ...
# def warp_fused_tensor_product( ... ): ...


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

    def report_time(dt):
        print(f"  time = {dt:.4e} s   TFLOPS = {FLOP / dt / 1e12:.3f}")

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
    tol = 1e-12

    # ── torch.einsum on GPU ──────────────────────────────────────────
    print("\n--- torch.einsum GPU ---")
    C_torch = torch_tensor_product(Ar_t, As_t, At_t, B_t)
    torch.cuda.synchronize()
    validate("torch.einsum", C_cpu, C_torch, tol)
    report_time(bench(torch_tensor_product, ntrys, Ar_t, As_t, At_t, B_t))

    # ── Triton kernels ───────────────────────────────────────────────
    if not HAS_TRITON:
        print("\nTriton not installed — skipping Triton benchmarks.")
        return

    print("\n--- Triton 3-stage ---")
    try:
        C_tri = triton_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()
        validate("Triton 3-stage", C_cpu, C_tri, tol)
        report_time(bench(triton_tensor_product, ntrys,
                          Ar_t, As_t, At_t, B_t, M, K, N))
    except Exception as e:
        print(f"  Triton compilation failed: {e}")
        return

    # ── Triton fused kernel ───────────────────────────────────────────
    print("\n--- Triton fused (single kernel, nw=4) ---")
    try:
        C_fused = triton_fused_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()
        validate("Triton fused", C_cpu, C_fused, tol)
        report_time(bench(triton_fused_tensor_product, ntrys,
                          Ar_t, As_t, At_t, B_t, M, K, N))
    except Exception as e:
        print(f"  Triton fused kernel failed: {e}")

    # ── Triton batched fused kernel ───────────────────────────────────
    print("\n--- Triton batched fused ---")
    try:
        C_bat = triton_batched_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
        torch.cuda.synchronize()
        validate("Triton batched", C_cpu, C_bat, tol)
        for batch_sz in [4, 8]:
            dt = bench(triton_batched_tensor_product, ntrys,
                       Ar_t, As_t, At_t, B_t, M, K, N, batch_sz, 4)
            print(f"  BATCH={batch_sz} nw=4: "
                  f"time={dt:.4e} s  TFLOPS={FLOP/dt/1e12:.3f}")
    except Exception as e:
        print(f"  Triton batched kernel failed: {e}")


if __name__ == '__main__':
    main()
