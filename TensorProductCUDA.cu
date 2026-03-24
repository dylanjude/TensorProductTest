// =====================================================================
// Native CUDA fused tensor-product kernel
// C[n, a, b, c] = sum_i Ar[a,i] * sum_j As[b,j] * sum_k At[c,k] * B[n,i,j,k]
//
// One block per batch element.  K*K*M threads per block.
// Operators and B loaded into shared memory; intermediates (tmpK, tmpJ)
// live entirely in shared memory with __syncthreads() between stages.
// =====================================================================

#include "TensorProductCUDA.hpp"
#include <cstdio>

// ── Templated kernel ────────────────────────────────────────────────

template <int M, int K>
__global__ void fused_tp_kernel(
    const double* __restrict__ Ar,
    const double* __restrict__ As,
    const double* __restrict__ At,
    const double* __restrict__ B,
    double* __restrict__ C,
    int N, int LDB, int LDC)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;

    constexpr int MK  = M * K;
    constexpr int K3  = K * K * K;
    constexpr int S1  = K * K * M;     // stage 1 output count
    constexpr int S2  = K * M * M;     // stage 2 output count
    constexpr int S3  = M * M * M;     // stage 3 output count
    constexpr int TPB = K * K * M;     // threads per block = S1
    constexpr int NPER_S2 = (S2 + TPB - 1) / TPB;
    constexpr int NPER_S3 = (S3 + TPB - 1) / TPB;

    __shared__ double sAt[MK], sAs[MK], sAr[MK];
    __shared__ double sB[K3];
    __shared__ double tmpK[S1];
    __shared__ double tmpJ[S2];

    // ── Load operators (M*K elements, cooperative) ──────────────────
    // Operators are column-major: A(m, k) = A[k * M + m]
    if (tid < MK) {
        sAr[tid] = Ar[tid];
        sAs[tid] = As[tid];
        sAt[tid] = At[tid];
    }

    // ── Load B[n] slice (K^3 elements) ──────────────────────────────
    // B is column-major: B(i, j, k) = B[n*LDB + i + K*j + K*K*k]
    if (tid < K3) {
        sB[tid] = B[n * LDB + tid];
    }

    __syncthreads();

    // ── Stage 1: tmpK[i,j,m] = sum_k At[m,k] * B[i,j,k] ───────────
    // tid maps to (i, j, m) with m fastest-varying
    // Operators are column-major: A(m, k) = A[k * M + m]
    {
        const int m = tid % M;
        const int j = (tid / M) % K;
        const int i = tid / (M * K);
        double acc = 0.0;
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            acc += sAt[kk * M + m] * sB[i + K * j + K * K * kk];
        }
        tmpK[tid] = acc;   // layout: tmpK[i * K*M + j * M + m]
    }

    __syncthreads();

    // ── Stage 2: tmpJ[i,b,c] = sum_j As[b,j] * tmpK[i,j,c] ────────
    #pragma unroll
    for (int t = 0; t < NPER_S2; ++t) {
        const int idx = tid + t * TPB;
        if (idx < S2) {
            const int c = idx % M;
            const int b = (idx / M) % M;
            const int i = idx / (M * M);
            double acc = 0.0;
            #pragma unroll
            for (int jj = 0; jj < K; ++jj) {
                acc += sAs[jj * M + b] * tmpK[i * K * M + jj * M + c];
            }
            tmpJ[idx] = acc;   // layout: tmpJ[i * M*M + b * M + c]
        }
    }

    __syncthreads();

    // ── Stage 3: C[n,a,b,c] = sum_i Ar[a,i] * tmpJ[i,b,c] ─────────
    // Write directly to global memory C with caller's LDC stride
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
                acc += sAr[ii * M + a] * tmpJ[ii * M * M + b * M + c];
            }
            C[n * LDC + a + M * b + M * M * c] = acc;
        }
    }
}

// ── Dispatch wrapper ────────────────────────────────────────────────

void launchFusedTPKernel(
    const double* d_Ar, const double* d_As, const double* d_At,
    const double* d_B, double* d_C,
    int M, int K, int N, int LDB, int LDC)
{
    const int TPB = K * K * M;

    #define DISPATCH(MM, KK) \
        if (M == MM && K == KK) { \
            fused_tp_kernel<MM, KK><<<N, TPB>>>( \
                d_Ar, d_As, d_At, d_B, d_C, N, LDB, LDC); \
            return; \
        }

    DISPATCH(8, 4)
    else DISPATCH(4, 8)
    else DISPATCH(4, 4)
    else DISPATCH(6, 3)
    else DISPATCH(3, 6)
    else DISPATCH(8, 8)
    else DISPATCH(10, 5)
    else DISPATCH(5, 10)
    else DISPATCH(12, 6)
    else DISPATCH(6, 12)
    else {
        fprintf(stderr,
            "ERROR: fused_tp_kernel has no template instantiation for M=%d, K=%d\n"
            "Add DISPATCH(%d, %d) to TensorProductCUDA.cu\n",
            M, K, M, K);
    }
    #undef DISPATCH
}
