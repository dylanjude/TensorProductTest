// =====================================================================
// FP64 Tensor Core fused tensor-product kernel (general M <= 8, K <= 8)
// C[n, a, b, c] = sum_i Ar[a,i] * sum_j As[b,j] * sum_k At[c,k] * B[n,i,j,k]
//
// Uses mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 PTX instruction.
// Requires SM >= 8.0 (Ampere / Hopper).
//
// One warp (32 threads) per batch element n.
// Operators padded to MPAD=8 rows and loaded into registers.
// For K > 4, contractions are split into ceil(K/4) MMA chunks with
// accumulation.  Intermediates reside in per-warp shared memory.
// sB (stage 1) and tmpJ (stages 2-3) share the same memory since
// their lifetimes do not overlap.
//
// Three sum-factorized stages, each using MMA instructions:
//   Stage 1: T1[m,j,i]   = sum_k At[m,k] * B[i,j,k]
//   Stage 2: T2[b,c,i]   = sum_j As[b,j] * T1[c,j,i]
//   Stage 3: C[a,b,c]    = sum_i Ar[a,i] * T2[b,c,i]
//
// Bank-conflict reduction via XOR permutation (Cui 2024, Fig. 4):
//   tmpK stored at (m ^ j)      + MPAD*j + MPAD*K*i
//   tmpJ stored at (c ^ b ^ i)  + MPAD*b + MPAD*MPAD*i
// =====================================================================

#include "TensorProductCUDA_TC.hpp"
#include <cstdio>

static constexpr int WARPS_PER_BLOCK_TC = 4;

template <int M, int K>
__global__ void fused_tp_tc_kernel(
    const double* __restrict__ Ar,
    const double* __restrict__ As,
    const double* __restrict__ At,
    const double* __restrict__ B,
    double* __restrict__ C,
    int N, int LDB, int LDC)
{
    static_assert(M >= 1 && M <= 8, "TC kernel requires 1 <= M <= 8");
    static_assert(K >= 1 && K <= 8, "TC kernel requires 1 <= K <= 8");

    constexpr int MPAD      = 8;                          // MMA row dimension
    constexpr int K_CHUNKS   = (K + 3) / 4;               // ceil(K/4)
    constexpr int MK         = M * K;                      // operator elements
    constexpr int K3         = K * K * K;                  // B slice size
    constexpr int KKM        = MPAD * K * K;               // tmpK size
    constexpr int KMM        = MPAD * MPAD * K;            // tmpJ size
    constexpr int SB_TMPJ    = (K3 > KMM) ? K3 : KMM;     // shared region for sB/tmpJ overlay
    constexpr int WS_PER_WARP = SB_TMPJ + KKM;             // total per-warp workspace
    constexpr int NUM_BATCHES_S1 = (K * K + 7) / 8;        // stage 1 batches

    const int warpId = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;
    const int n      = blockIdx.x * WARPS_PER_BLOCK_TC + warpId;

    // ── Shared memory layout ───────────────────────────────────────────
    // Operators: MPAD-strided, zero-padded for m >= M (3 * MPAD * K)
    __shared__ double sOps[3 * MPAD * K];
    double* sAt = sOps;
    double* sAs = sOps + MPAD * K;
    double* sAr = sOps + 2 * MPAD * K;

    // Per-warp workspace: sB/tmpJ overlay + tmpK
    __shared__ double ws[WARPS_PER_BLOCK_TC][WS_PER_WARP];
    double* sB   = ws[warpId];              // valid during stage 1
    double* tmpJ = ws[warpId];              // valid during stages 2-3 (same memory as sB)
    double* tmpK = ws[warpId] + SB_TMPJ;    // valid during stages 1-2

    // ── Load operators into MPAD-strided shared memory ─────────────────
    // Store as sOp[k * MPAD + m], zero-padding rows m >= M
    {
        const int tid   = threadIdx.x;
        const int total = WARPS_PER_BLOCK_TC * 32;
        for (int idx = tid; idx < 3 * MPAD * K; idx += total) {
            const int op  = idx / (MPAD * K);   // 0=At, 1=As, 2=Ar
            const int rem = idx % (MPAD * K);
            const int k   = rem / MPAD;
            const int m   = rem % MPAD;
            const double* src = (op == 0) ? At : (op == 1) ? As : Ar;
            sOps[idx] = (m < M && k < K) ? src[k * M + m] : 0.0;
        }
    }

    // Load B slice into per-warp shared memory
    if (n < N) {
        for (int t = lane; t < K3; t += 32) {
            sB[t] = B[(size_t)n * LDB + t];
        }
    }

    __syncthreads();  // ensure operators visible; all warps participate

    if (n >= N) return;  // early exit AFTER syncthreads

    // ── Pre-load operator A-fragments for each K-chunk ─────────────────
    // MMA A is row-major 8×4: thread lane holds A[lane/4][lane%4]
    // Shared operators stored as Op[k * MPAD + m]
    double a_At[K_CHUNKS], a_As[K_CHUNKS], a_Ar[K_CHUNKS];
    #pragma unroll
    for (int kc = 0; kc < K_CHUNKS; ++kc) {
        const int k_col = kc * 4 + (lane % 4);
        a_At[kc] = (k_col < K) ? sAt[k_col * MPAD + (lane / 4)] : 0.0;
        a_As[kc] = (k_col < K) ? sAs[k_col * MPAD + (lane / 4)] : 0.0;
        a_Ar[kc] = (k_col < K) ? sAr[k_col * MPAD + (lane / 4)] : 0.0;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Stage 1: tmpK[(m^j) + MPAD*j + MPAD*K*i] = sum_k At[m,k] * B[i+K*j+K²*k]
    //   Batches K² output (i,j) pairs into groups of 8.
    //   Accumulates across ceil(K/4) MMA chunks for the k contraction.
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int batch = 0; batch < NUM_BATCHES_S1; ++batch) {
        double d0 = 0.0, d1 = 0.0;

        #pragma unroll
        for (int kc = 0; kc < K_CHUNKS; ++kc) {
            const int n_col    = lane / 4;
            const int k_local  = lane % 4;
            const int k_global = kc * 4 + k_local;
            const int linear   = 8 * batch + n_col;
            const int i_idx    = linear / K;
            const int j_idx    = linear % K;

            const double b_val = (linear < K * K && k_global < K)
                ? sB[i_idx + K * j_idx + K * K * k_global] : 0.0;

            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=d"(d0), "=d"(d1)
                : "d"(a_At[kc]), "d"(b_val), "d"(d0), "d"(d1)
            );
        }

        // Store results — guard against out-of-range (i,j) pairs
        const int m    = lane / 4;
        const int col0 = (lane % 4) * 2;
        const int col1 = col0 + 1;
        const int lin0 = 8 * batch + col0;
        const int lin1 = 8 * batch + col1;

        if (lin0 < K * K) {
            const int i0 = lin0 / K, j0 = lin0 % K;
            tmpK[(m ^ j0) + MPAD * j0 + MPAD * K * i0] = d0;
        }
        if (lin1 < K * K) {
            const int i1 = lin1 / K, j1 = lin1 % K;
            tmpK[(m ^ j1) + MPAD * j1 + MPAD * K * i1] = d1;
        }
    }

    __syncwarp();

    // ═══════════════════════════════════════════════════════════════════
    // Stage 2: tmpJ[(c^b^ii) + MPAD*b + MPAD²*ii]
    //        = sum_j As[b,j] * tmpK[(c^j) + MPAD*j + MPAD*K*ii]
    //   One MMA per (ii, K-chunk) pair.  Accumulates across K-chunks.
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int ii = 0; ii < K; ++ii) {
        double d0 = 0.0, d1 = 0.0;

        #pragma unroll
        for (int kc = 0; kc < K_CHUNKS; ++kc) {
            const int c_rd     = lane / 4;
            const int j_local  = lane % 4;
            const int j_actual = kc * 4 + j_local;

            const double b_val = (j_actual < K)
                ? tmpK[(c_rd ^ j_actual) + MPAD * j_actual + MPAD * K * ii] : 0.0;

            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=d"(d0), "=d"(d1)
                : "d"(a_As[kc]), "d"(b_val), "d"(d0), "d"(d1)
            );
        }

        const int b_idx  = lane / 4;
        const int c0_idx = (lane % 4) * 2;
        const int c1_idx = c0_idx + 1;
        tmpJ[(c0_idx ^ b_idx ^ ii) + MPAD * b_idx + MPAD * MPAD * ii] = d0;
        tmpJ[(c1_idx ^ b_idx ^ ii) + MPAD * b_idx + MPAD * MPAD * ii] = d1;
    }

    __syncwarp();

    // ═══════════════════════════════════════════════════════════════════
    // Stage 3: C[n*LDC + a + M*b + M²*cc]
    //        = sum_i Ar[a,i] * tmpJ[(cc^b^i) + MPAD*b + MPAD²*i]
    //   One MMA per (cc, K-chunk) pair.  Accumulates across K-chunks.
    //   Bounds-check C writes for M < 8.
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int cc = 0; cc < M; ++cc) {
        double d0 = 0.0, d1 = 0.0;

        #pragma unroll
        for (int kc = 0; kc < K_CHUNKS; ++kc) {
            const int b_rd     = lane / 4;
            const int i_local  = lane % 4;
            const int i_actual = kc * 4 + i_local;

            const double b_val = (i_actual < K)
                ? tmpJ[(cc ^ b_rd ^ i_actual) + MPAD * b_rd + MPAD * MPAD * i_actual] : 0.0;

            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=d"(d0), "=d"(d1)
                : "d"(a_Ar[kc]), "d"(b_val), "d"(d0), "d"(d1)
            );
        }

        const int a_idx  = lane / 4;
        const int b0_idx = (lane % 4) * 2;
        const int b1_idx = b0_idx + 1;

        if (a_idx < M && b0_idx < M)
            C[(size_t)n * LDC + a_idx + M * b0_idx + M * M * cc] = d0;
        if (a_idx < M && b1_idx < M)
            C[(size_t)n * LDC + a_idx + M * b1_idx + M * M * cc] = d1;
    }
}

// ── Dispatch wrapper ────────────────────────────────────────────────

void launchFusedTPKernel_TC(
    const double* d_Ar, const double* d_As, const double* d_At,
    const double* d_B, double* d_C,
    int M, int K, int N, int LDB, int LDC)
{
    constexpr int WPB = WARPS_PER_BLOCK_TC;

    #define DISPATCH_TC(MM, KK) \
        if (M == MM && K == KK) { \
            const int blocks  = (N + WPB - 1) / WPB; \
            const int threads = WPB * 32; \
            fused_tp_tc_kernel<MM, KK><<<blocks, threads>>>( \
                d_Ar, d_As, d_At, d_B, d_C, N, LDB, LDC); \
            return; \
        }

    DISPATCH_TC(8, 4)
    else DISPATCH_TC(4, 8)
    else DISPATCH_TC(5, 3)
    else DISPATCH_TC(3, 5)
    else DISPATCH_TC(4, 4)
    else DISPATCH_TC(6, 3)
    else DISPATCH_TC(3, 6)
    else DISPATCH_TC(8, 8)
    else {
        fprintf(stderr,
            "ERROR: fused_tp_tc_kernel has no instantiation for M=%d, K=%d\n"
            "Add DISPATCH_TC(%d, %d) to TensorProductCUDA_TC.cu\n",
            M, K, M, K);
    }

    #undef DISPATCH_TC
}
