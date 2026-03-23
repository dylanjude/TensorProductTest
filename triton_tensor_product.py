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
    from torch.utils.cpp_extension import load_inline
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

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
# CUDA C++ fused kernel — shared memory + __syncthreads__ (mirrors OCCA)
# ─────────────────────────────────────────────────────────────────────
# Compiled via torch.utils.cpp_extension.load_inline (nvcc → cubin).
# One block per batch element, K*K*M threads per block.
# Operators and B loaded into shared memory.  Intermediates (tmpK, tmpJ)
# live entirely in shared memory with __syncthreads() between stages.

_CUDA_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Template parameters for compile-time M, K
template <int M, int K>
__global__ void fused_tp_kernel(
    const double* __restrict__ Ar,
    const double* __restrict__ As,
    const double* __restrict__ At,
    const double* __restrict__ B,
    double* __restrict__ C,
    int N)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;

    constexpr int MK  = M * K;
    constexpr int K3  = K * K * K;
    constexpr int S1  = K * K * M;    // stage 1 outputs
    constexpr int S2  = K * M * M;    // stage 2 outputs
    constexpr int S3  = M * M * M;    // stage 3 outputs
    constexpr int TPB = K * K * M;    // threads per block = S1
    constexpr int NPER_S2 = (S2 + TPB - 1) / TPB;
    constexpr int NPER_S3 = (S3 + TPB - 1) / TPB;

    __shared__ double sAt[MK], sAs[MK], sAr[MK];
    __shared__ double sB[K3];
    __shared__ double tmpK[S1];
    __shared__ double tmpJ[S2];

    // ── Load operators (M*K elements, cooperative) ──────────────
    if (tid < MK) {
        sAr[tid] = Ar[tid];
        sAs[tid] = As[tid];
        sAt[tid] = At[tid];
    }

    // ── Load B[n] (K^3 elements) ────────────────────────────────
    if (tid < K3) {
        sB[tid] = B[n * K3 + tid];
    }

    __syncthreads();

    // ── Stage 1: tmpK[i,j,m] = sum_k At[m,k] * B[i,j,k] ──────
    {
        const int m  = tid % M;
        const int j  = (tid / M) % K;
        const int i  = tid / (M * K);
        double acc = 0.0;
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            acc += sAt[m * K + kk] * sB[i * K * K + j * K + kk];
        }
        tmpK[tid] = acc;  // layout: [i][j][m]
    }

    __syncthreads();

    // ── Stage 2: tmpJ[i,b,c] = sum_j As[b,j] * tmpK[i,j,c] ───
    #pragma unroll
    for (int t = 0; t < NPER_S2; ++t) {
        const int idx = tid + t * TPB;
        if (idx < S2) {
            const int c  = idx % M;
            const int b  = (idx / M) % M;
            const int i  = idx / (M * M);
            double acc = 0.0;
            #pragma unroll
            for (int jj = 0; jj < K; ++jj) {
                acc += sAs[b * K + jj] * tmpK[i * K * M + jj * M + c];
            }
            tmpJ[idx] = acc;  // layout: [i][b][c]
        }
    }

    __syncthreads();

    // ── Stage 3: C[n,a,b,c] = sum_i Ar[a,i] * tmpJ[i,b,c] ────
    #pragma unroll
    for (int t = 0; t < NPER_S3; ++t) {
        const int idx = tid + t * TPB;
        if (idx < S3) {
            const int c = idx % M;
            const int b = (idx / M) % M;
            const int a = idx / (M * M);
            double acc = 0.0;
            #pragma unroll
            for (int ii = 0; ii < K; ++ii) {
                acc += sAr[a * K + ii] * tmpJ[ii * M * M + b * M + c];
            }
            C[n * S3 + idx] = acc;
        }
    }
}

// ── Dispatch wrapper (selects template instantiation) ───────────
torch::Tensor fused_tp_forward(
    torch::Tensor Ar, torch::Tensor As, torch::Tensor At,
    torch::Tensor B, int M_val, int K_val)
{
    const int N = B.size(0);
    auto C = torch::empty({N, M_val, M_val, M_val},
                          B.options().dtype(torch::kFloat64));
    const int TPB = K_val * K_val * M_val;

    // Template dispatch for common (M, K) pairs
    #define DISPATCH(MM, KK) \
        if (M_val == MM && K_val == KK) { \
            fused_tp_kernel<MM, KK><<<N, TPB>>>( \
                Ar.data_ptr<double>(), As.data_ptr<double>(), \
                At.data_ptr<double>(), B.data_ptr<double>(), \
                C.data_ptr<double>(), N); \
        }

    DISPATCH(8, 4)
    else DISPATCH(4, 8)
    else DISPATCH(6, 3)
    else DISPATCH(4, 4)
    else DISPATCH(8, 8)
    else DISPATCH(16, 8)
    else DISPATCH(3, 6)
    else DISPATCH(10, 5)
    else {
        TORCH_CHECK(false,
            "fused_tp_kernel: no template instantiation for M=",
            M_val, " K=", K_val,
            ". Add DISPATCH(", M_val, ", ", K_val, ") to the source.");
    }
    #undef DISPATCH

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_tp_forward, "Fused tensor product (CUDA)");
}
"""

_cuda_ext = None

def _get_cuda_ext():
    """Lazily compile the CUDA extension on first use."""
    global _cuda_ext
    if _cuda_ext is None:
        print("  Compiling CUDA C++ fused kernel (first run only)...")
        _cuda_ext = load_inline(
            name='fused_tp',
            cpp_sources='',
            cuda_sources=_CUDA_KERNEL_SRC,
            functions=['forward'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return _cuda_ext


def cuda_fused_tensor_product(Ar, As, At, B, M, K):
    """Run the CUDA C++ fused kernel.  All tensors must be contiguous on CUDA."""
    ext = _get_cuda_ext()
    return ext.forward(Ar.contiguous(), As.contiguous(),
                       At.contiguous(), B.contiguous(), M, K)


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

    # ── CUDA C++ fused kernel (shared mem + syncthreads) ─────────────
    if not HAS_CUDA_EXT:
        print("\ntorch.utils.cpp_extension not available — skipping CUDA fused kernel.")
    else:
        print("\n--- CUDA C++ fused (shared mem + syncthreads) ---")
        try:
            C_fused = cuda_fused_tensor_product(Ar_t, As_t, At_t, B_t, M, K)
            torch.cuda.synchronize()
            validate("CUDA fused", C_cpu, C_fused, tol)
            report_time(bench(cuda_fused_tensor_product, ntrys,
                              Ar_t, As_t, At_t, B_t, M, K))
        except Exception as e:
            print(f"  CUDA fused kernel failed: {e}")


if __name__ == '__main__':
    main()
