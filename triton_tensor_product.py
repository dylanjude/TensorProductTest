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

try:
    import warp as wp
    wp.init()
    HAS_WARP = True
except (ImportError, Exception):
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


def bench(fn, ntrys, *args):
    """Warm up (3 iters), then time ntrys iterations.  Returns seconds/iter."""
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ntrys):
        fn(*args)
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
# CUDA graph wrapper for Triton 3-stage
# ─────────────────────────────────────────────────────────────────────

class TritonGraphed:
    """Captures the 3-stage Triton launch sequence as a CUDA graph."""

    def __init__(self, Ar, As, At, B, M, K, N):
        BLOCK1 = next_power_of_2(K * K * M)
        BLOCK2 = next_power_of_2(K * M * M)
        BLOCK3 = next_power_of_2(M * M * M)
        grid = (N,)

        # Pre-allocate all buffers (reused across replays)
        self.tmp1 = torch.empty((N, K, K, M), dtype=torch.float64, device='cuda')
        self.tmp2 = torch.empty((N, K, M, M), dtype=torch.float64, device='cuda')
        self.C    = torch.empty((N, M, M, M), dtype=torch.float64, device='cuda')

        # Warm up the kernels before capture
        _stage1_kernel[grid](At, B, self.tmp1, N, M, K, BLOCK1)
        _stage2_kernel[grid](As, self.tmp1, self.tmp2, N, M, K, BLOCK2)
        _stage3_kernel[grid](Ar, self.tmp2, self.C, N, M, K, BLOCK3)
        torch.cuda.synchronize()

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            _stage1_kernel[grid](At, B, self.tmp1, N, M, K, BLOCK1)
            _stage2_kernel[grid](As, self.tmp1, self.tmp2, N, M, K, BLOCK2)
            _stage3_kernel[grid](Ar, self.tmp2, self.C, N, M, K, BLOCK3)

    def replay(self):
        self._graph.replay()
        return self.C


# ─────────────────────────────────────────────────────────────────────
# NVIDIA Warp — 3-stage sum-factorised tensor product (simple)
# ─────────────────────────────────────────────────────────────────────
# One Warp thread per (n, output-element) pair in each stage.
# No shared memory — every thread reads operators from global memory
# (L1/L2 cached).  Three separate kernel launches.

if HAS_WARP:

    @wp.kernel
    def _warp_stage1(
        At: wp.array2d(dtype=wp.float64),   # (M, K)
        B: wp.array2d(dtype=wp.float64),     # (N, K^3)
        out: wp.array2d(dtype=wp.float64),   # (N, K*K*M)
        M_val: int, K_val: int,
    ):
        """out[n, i*K*M + j*M + m] = sum_k At[m,k] * B[n, i*K*K + j*K + k]"""
        tid = wp.tid()
        KKM = K_val * K_val * M_val
        n = tid // KKM
        rem = tid - n * KKM
        m = rem % M_val
        tmp = rem // M_val
        j = tmp % K_val
        i = tmp // K_val
        acc = wp.float64(0.0)
        for kk in range(K_val):
            acc += At[m, kk] * B[n, i * K_val * K_val + j * K_val + kk]
        out[n, rem] = acc

    @wp.kernel
    def _warp_stage2(
        As: wp.array2d(dtype=wp.float64),   # (M, K)
        inp: wp.array2d(dtype=wp.float64),  # (N, K*K*M)
        out: wp.array2d(dtype=wp.float64),  # (N, K*M*M)
        M_val: int, K_val: int,
    ):
        """out[n, i*M*M + b*M + c] = sum_j As[b,j] * inp[n, i*K*M + j*M + c]"""
        tid = wp.tid()
        KMM = K_val * M_val * M_val
        n = tid // KMM
        rem = tid - n * KMM
        c = rem % M_val
        tmp = rem // M_val
        b = tmp % M_val
        i = tmp // M_val
        acc = wp.float64(0.0)
        for jj in range(K_val):
            acc += As[b, jj] * inp[n, i * K_val * M_val + jj * M_val + c]
        out[n, rem] = acc

    @wp.kernel
    def _warp_stage3(
        Ar: wp.array2d(dtype=wp.float64),   # (M, K)
        inp: wp.array2d(dtype=wp.float64),  # (N, K*M*M)
        out: wp.array2d(dtype=wp.float64),  # (N, M*M*M)
        M_val: int, K_val: int,
    ):
        """out[n, a*M*M + b*M + c] = sum_i Ar[a,i] * inp[n, i*M*M + b*M + c]"""
        tid = wp.tid()
        MMM = M_val * M_val * M_val
        n = tid // MMM
        rem = tid - n * MMM
        c = rem % M_val
        tmp = rem // M_val
        b = tmp % M_val
        a = tmp // M_val
        acc = wp.float64(0.0)
        for ii in range(K_val):
            acc += Ar[a, ii] * inp[n, ii * M_val * M_val + b * M_val + c]
        out[n, rem] = acc


def warp_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N):
    """Run the 3-stage Warp kernel.  All tensors must be on CUDA (torch)."""
    At_w = wp.from_torch(At_t.contiguous().reshape(M, K))
    As_w = wp.from_torch(As_t.contiguous().reshape(M, K))
    Ar_w = wp.from_torch(Ar_t.contiguous().reshape(M, K))
    B_w  = wp.from_torch(B_t.contiguous().reshape(N, K * K * K))

    tmp1_t = torch.empty((N, K * K * M), dtype=torch.float64, device='cuda')
    tmp2_t = torch.empty((N, K * M * M), dtype=torch.float64, device='cuda')
    C_t    = torch.empty((N, M * M * M), dtype=torch.float64, device='cuda')

    tmp1_w = wp.from_torch(tmp1_t)
    tmp2_w = wp.from_torch(tmp2_t)
    C_w    = wp.from_torch(C_t)

    wp.launch(_warp_stage1, dim=N * K * K * M,
              inputs=[At_w, B_w, tmp1_w, M, K])
    wp.launch(_warp_stage2, dim=N * K * M * M,
              inputs=[As_w, tmp1_w, tmp2_w, M, K])
    wp.launch(_warp_stage3, dim=N * M * M * M,
              inputs=[Ar_w, tmp2_w, C_w, M, K])

    return C_t.reshape(N, M, M, M)


# ─────────────────────────────────────────────────────────────────────
# NVIDIA Warp — tile_matmul fused kernel (M=8, K=4)
# ─────────────────────────────────────────────────────────────────────
# Expresses each contraction stage as a cooperative tile_matmul, which
# has proper __syncthreads() internally.  One block per batch element.
#
# Stage 1: T1[K²,M] = B_reshaped[K²,K] @ At^T[K,M]      (1 matmul)
# Stage 2: T2_i[M,M] = As[M,K] @ T1_i[K,M] for each i   (K matmuls)
# Stage 3: C[M,M²]  = Ar[M,K] @ T2_view[K,M²]           (1 matmul)
#
# Intermediates (T1, T2) go through global memory (L2-cached).
# T2 is passed as two array views of the same memory:
#   T2_w  [N, K*M, M] = [N, 32, 8]  for Stage 2 writes
#   T2v_w [N, K, M²]  = [N, 4, 64]  for Stage 3 reads

if HAS_WARP:

    # Compile-time tile shape constants for M=8, K=4
    _TM = wp.constant(8)     # M
    _TK = wp.constant(4)     # K
    _TK2 = wp.constant(16)   # K² = 16
    _TM2 = wp.constant(64)   # M² = 64

    @wp.kernel
    def _warp_tiled_84(
        At_T:  wp.array2d(dtype=wp.float64),   # [K, M]    = [4, 8]  (transposed)
        As:    wp.array2d(dtype=wp.float64),    # [M, K]    = [8, 4]
        Ar:    wp.array2d(dtype=wp.float64),    # [M, K]    = [8, 4]
        B:     wp.array3d(dtype=wp.float64),    # [N, K², K]  = [N, 16, 4]
        T1:    wp.array3d(dtype=wp.float64),    # [N, K², M]  = [N, 16, 8]
        T2:    wp.array3d(dtype=wp.float64),    # [N, K*M, M] = [N, 32, 8]
        T2v:   wp.array3d(dtype=wp.float64),    # [N, K, M²]  = [N, 4, 64]  (same mem as T2)
        C:     wp.array3d(dtype=wp.float64),    # [N, M, M²]  = [N, 8, 64]
    ):
        n = wp.tid()

        # ── Stage 1: T1[K²,M] = B[K²,K] @ At^T[K,M] ──────────────
        b_tile  = wp.tile_load(B[n],  shape=(_TK2, _TK))   # [16, 4]
        at_tile = wp.tile_load(At_T,  shape=(_TK, _TM))    # [4, 8]
        t1_tile = wp.tile_matmul(b_tile, at_tile)           # [16, 8]
        wp.tile_store(T1[n], t1_tile)

        # ── Stage 2: T2_i[M,M] = As[M,K] @ T1_i[K,M]  (4 slices) ─
        as_tile = wp.tile_load(As, shape=(_TM, _TK))       # [8, 4]
        for i in range(4):
            t1i = wp.tile_load(T1[n], shape=(_TK, _TM),
                               offset=(i * _TK, 0))        # [4, 8]
            t2i = wp.tile_matmul(as_tile, t1i)              # [8, 8]
            wp.tile_store(T2[n], t2i,
                          offset=(i * _TM, 0))              # rows [i*8 .. i*8+8]

        # ── Stage 3: C[M,M²] = Ar[M,K] @ T2v[K,M²] ───────────────
        ar_tile  = wp.tile_load(Ar,     shape=(_TM, _TK))  # [8, 4]
        t2v_tile = wp.tile_load(T2v[n], shape=(_TK, _TM2)) # [4, 64]
        c_tile   = wp.tile_matmul(ar_tile, t2v_tile)        # [8, 64]
        wp.tile_store(C[n], c_tile)


def warp_fused_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N):
    """Run the tile_matmul fused Warp kernel.  Currently only M=8, K=4."""
    if M != 8 or K != 4:
        raise ValueError(f"Fused Warp kernel only supports M=8, K=4 (got M={M}, K={K})")

    # Pre-transpose At for Stage 1: need [K, M] = [4, 8]
    At_T_t = At_t.t().contiguous()

    At_T_w = wp.from_torch(At_T_t)                                      # [4, 8]
    As_w   = wp.from_torch(As_t.contiguous())                            # [8, 4]
    Ar_w   = wp.from_torch(Ar_t.contiguous())                            # [8, 4]
    B_w    = wp.from_torch(B_t.contiguous().reshape(N, K*K, K))          # [N, 16, 4]

    T1_t = torch.empty((N, K*K, M), dtype=torch.float64, device='cuda') # [N, 16, 8]
    T2_t = torch.empty((N, K*M, M), dtype=torch.float64, device='cuda') # [N, 32, 8]
    C_t  = torch.empty((N, M, M*M), dtype=torch.float64, device='cuda') # [N, 8, 64]

    # T2 viewed as [N, K, M²] for Stage 3 — same memory, different shape
    T2v_t = T2_t.reshape(N, K, M*M)

    T1_w  = wp.from_torch(T1_t)
    T2_w  = wp.from_torch(T2_t)
    T2v_w = wp.from_torch(T2v_t)
    C_w   = wp.from_torch(C_t)

    wp.launch_tiled(_warp_tiled_84, dim=[N],
                    inputs=[At_T_w, As_w, Ar_w, B_w, T1_w, T2_w, T2v_w, C_w],
                    block_dim=32)

    return C_t.reshape(N, M, M, M)


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

    # ── torch.compile'd einsum ───────────────────────────────────────
    print("\n--- torch.compile(einsum) GPU ---")
    try:
        # Compilation happens on first call; run a few warm-ups
        for _ in range(3):
            torch_tensor_product_compiled(Ar_t, As_t, At_t, B_t)
        torch.cuda.synchronize()
        C_compiled = torch_tensor_product_compiled(Ar_t, As_t, At_t, B_t)
        torch.cuda.synchronize()
        validate("torch.compile", C_cpu, C_compiled, tol)
        report_time(bench(torch_tensor_product_compiled, ntrys,
                          Ar_t, As_t, At_t, B_t))
    except Exception as e:
        print(f"  torch.compile failed: {e}")

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
        print(f"  Triton compilation failed (likely SM < 7.0): {e}")
        return

    # ── CUDA graph of Triton 3-stage ─────────────────────────────────
    print("\n--- Triton 3-stage + CUDA graph ---")
    try:
        graphed = TritonGraphed(Ar_t, As_t, At_t, B_t, M, K, N)
        C_graphed = graphed.replay()
        torch.cuda.synchronize()
        validate("Triton+CUDAgraph", C_cpu, C_graphed, tol)
        report_time(bench(graphed.replay, ntrys))
    except Exception as e:
        print(f"  CUDA graph capture failed: {e}")

    # ── NVIDIA Warp 3-stage kernel ───────────────────────────────────
    if not HAS_WARP:
        print("\nwarp-lang not installed — skipping Warp benchmarks.")
    else:
        print("\n--- NVIDIA Warp 3-stage ---")
        try:
            C_warp = warp_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
            torch.cuda.synchronize()
            validate("Warp 3-stage", C_cpu, C_warp, tol)
            report_time(bench(warp_tensor_product, ntrys,
                              Ar_t, As_t, At_t, B_t, M, K, N))
        except Exception as e:
            print(f"  Warp kernel failed: {e}")

        # ── NVIDIA Warp tile_matmul fused kernel (M=8 K=4 only) ───────
        if M == 8 and K == 4:
            print("\n--- NVIDIA Warp tile_matmul fused ---")
            try:
                C_wfused = warp_fused_tensor_product(Ar_t, As_t, At_t, B_t, M, K, N)
                torch.cuda.synchronize()
                validate("Warp tile_matmul", C_cpu, C_wfused, tol)
                report_time(bench(warp_fused_tensor_product, ntrys,
                                  Ar_t, As_t, At_t, B_t, M, K, N))
            except Exception as e:
                print(f"  Warp tile_matmul kernel failed: {e}")


if __name__ == '__main__':
    main()
