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

try:
    from numba import cuda, float64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

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
# Numba CUDA kernel — shared memory, syncthreads (mirrors OCCA)
# ─────────────────────────────────────────────────────────────────────
# One block per batch element, K*K*M threads per block.
# Operators and B loaded into shared memory.  Intermediates (tmpK, tmpJ)
# live entirely in shared memory with syncthreads between stages.
# Each thread handles multiple outputs in stages 2 and 3.

def make_numba_tp_kernel(M, K):
    """Build a Numba CUDA kernel specialised for given M, K."""
    TPB = K * K * M          # threads per block (128 for M=8, K=4)
    S1  = K * K * M          # stage 1 outputs per slice
    S2  = K * M * M          # stage 2 outputs per slice
    S3  = M * M * M          # stage 3 outputs per slice
    MK  = M * K              # operator size
    K3  = K * K * K           # B slice size
    nper_s2 = (S2 + TPB - 1) // TPB   # outputs per thread, stage 2
    nper_s3 = (S3 + TPB - 1) // TPB   # outputs per thread, stage 3

    @cuda.jit
    def _kernel(Ar, As, At, B, C, N):
        n = cuda.blockIdx.x
        if n >= N:
            return
        tid = cuda.threadIdx.x

        # ── shared memory ────────────────────────────────────────────
        sAt = cuda.shared.array(shape=(MK,), dtype=float64)
        sAs = cuda.shared.array(shape=(MK,), dtype=float64)
        sAr = cuda.shared.array(shape=(MK,), dtype=float64)
        sB  = cuda.shared.array(shape=(K3,), dtype=float64)
        tmpK = cuda.shared.array(shape=(S1,), dtype=float64)
        tmpJ = cuda.shared.array(shape=(S2,), dtype=float64)

        # ── load operators (M*K = 32 elements) ──────────────────────
        if tid < MK:
            m = tid // K
            k = tid % K
            sAr[tid] = Ar[m, k]
            sAs[tid] = As[m, k]
            sAt[tid] = At[m, k]

        # ── load B[n] (K^3 = 64 elements) ───────────────────────────
        if tid < K3:
            sB[tid] = B[n, tid // (K * K), (tid // K) % K, tid % K]

        cuda.syncthreads()

        # ── Stage 1: tmpK[i,j,m] = sum_kk At[m,kk] * B[i,j,kk] ────
        # S1 = TPB so exactly one output per thread
        m  = tid % M
        j1 = (tid // M) % K
        i1 = tid // (M * K)
        acc = 0.0
        for kk in range(K):
            acc += sAt[m * K + kk] * sB[i1 * K * K + j1 * K + kk]
        tmpK[tid] = acc

        cuda.syncthreads()

        # ── Stage 2: tmpJ[i,b,c] = sum_jj As[b,jj] * tmpK[i,jj,c] ─
        for t in range(nper_s2):
            idx = tid + t * TPB
            if idx < S2:
                c  = idx % M
                b  = (idx // M) % M
                i2 = idx // (M * M)
                acc = 0.0
                for jj in range(K):
                    acc += sAs[b * K + jj] * tmpK[i2 * K * M + jj * M + c]
                tmpJ[idx] = acc

        cuda.syncthreads()

        # ── Stage 3: C[n,a,b,c] = sum_ii Ar[a,ii] * tmpJ[ii,b,c] ──
        for t in range(nper_s3):
            idx = tid + t * TPB
            if idx < S3:
                c = idx % M
                b = (idx // M) % M
                a = idx // (M * M)
                acc = 0.0
                for ii in range(K):
                    acc += sAr[a * K + ii] * tmpJ[ii * M * M + b * M + c]
                C[n, a, b, c] = acc

    return _kernel, TPB


def numba_tensor_product(Ar, As, At, B, M, K, N):
    """Run the Numba CUDA fused kernel.  Inputs are numpy arrays on host;
    returns a torch CUDA tensor for consistency with other benchmarks."""
    kernel, TPB = make_numba_tp_kernel(M, K)

    # Transfer to device as contiguous float64 arrays
    d_Ar = cuda.to_device(np.ascontiguousarray(Ar))
    d_As = cuda.to_device(np.ascontiguousarray(As))
    d_At = cuda.to_device(np.ascontiguousarray(At))
    d_B  = cuda.to_device(np.ascontiguousarray(B))
    d_C  = cuda.device_array((N, M, M, M), dtype=np.float64)

    kernel[N, TPB](d_Ar, d_As, d_At, d_B, d_C, N)
    cuda.synchronize()

    # Wrap result as a torch tensor (zero-copy via __cuda_array_interface__)
    return torch.as_tensor(d_C, device='cuda')


class NumbaGraphed:
    """Pre-compiled Numba kernel with pre-allocated device arrays for timing."""

    def __init__(self, Ar, As, At, B, M, K, N):
        self.kernel, self.TPB = make_numba_tp_kernel(M, K)
        self.N = N
        self.d_Ar = cuda.to_device(np.ascontiguousarray(Ar))
        self.d_As = cuda.to_device(np.ascontiguousarray(As))
        self.d_At = cuda.to_device(np.ascontiguousarray(At))
        self.d_B  = cuda.to_device(np.ascontiguousarray(B))
        self.d_C  = cuda.device_array((N, M, M, M), dtype=np.float64)
        # Warm up (JIT compile)
        self.kernel[N, self.TPB](self.d_Ar, self.d_As, self.d_At,
                                  self.d_B, self.d_C, N)
        cuda.synchronize()

    def run(self):
        self.kernel[self.N, self.TPB](
            self.d_Ar, self.d_As, self.d_At, self.d_B, self.d_C, self.N)

    def get_C(self):
        return torch.as_tensor(self.d_C, device='cuda')


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

    # ── Numba CUDA kernel ────────────────────────────────────────────
    if not HAS_NUMBA:
        print("\nNumba not installed — skipping Numba benchmark.")
        print("  Install with: pip install numba")
    else:
        print("\n--- Numba CUDA (shared mem + syncthreads) ---")
        try:
            C_numba = numba_tensor_product(Ar, As, At, B, M, K, N)
            validate("Numba CUDA", C_cpu, C_numba, tol)

            # For fair timing, use pre-allocated device arrays
            nb = NumbaGraphed(Ar, As, At, B, M, K, N)
            report_time(bench(nb.run, ntrys))
        except Exception as e:
            print(f"  Numba CUDA failed: {e}")


if __name__ == '__main__':
    main()
