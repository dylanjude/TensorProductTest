// =====================================================================
// FP64 Tensor Core fused tensor-product kernel
// C[n, a, b, c] = sum_i Ar[a,i] * sum_j As[b,j] * sum_k At[c,k] * B[n,i,j,k]
//
// Uses mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 PTX instruction.
// Requires SM >= 8.0 (Ampere / Hopper).
//
// One warp (32 threads) per batch element n.
// Operators pre-loaded into registers; B and intermediates in shared memory.
// Three sum-factorized stages, each using MMA instructions:
//   Stage 1: T1[m,j,i]   = sum_k At[m,k] * B[i,j,k]       — 2 MMA ops
//   Stage 2: T2[b,c,i]   = sum_j As[b,j] * T1[c,j,i]      — K MMA ops
//   Stage 3: C[a,b,c]    = sum_i Ar[a,i] * T2[b,c,i]       — M MMA ops
//
// Bank-conflict reduction via XOR permutation (Cui 2024, Fig. 4):
//   tmpK stored at (m ^ j)  + M*j + M*K*i   instead of m + M*j + M*K*i
//   tmpJ stored at (c ^ b ^ i) + M*b + M*M*i instead of c + M*b + M*M*i
// This reduces stage 3 shared-memory bank conflicts from 16-way to 2-way
// (2-way is the theoretical minimum for 32 threads accessing M=8 doubles).
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
    static_assert(M == 8, "TC kernel currently requires M=8");
    static_assert(K == 4, "TC kernel currently requires K=4");

    constexpr int MK  = M * K;         // 32
    constexpr int K3  = K * K * K;     // 64
    constexpr int KKM = K * K * M;     // 128 (tmpK size)
    constexpr int KMM = K * M * M;     // 256 (tmpJ size)
    constexpr int NUM_BATCHES_S1 = (K * K + 7) / 8;  // 2

    const int warpId = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;
    const int n      = blockIdx.x * WARPS_PER_BLOCK_TC + warpId;

    // ── Shared memory layout ───────────────────────────────────────────
    // Operators: shared across all warps (3 * MK = 96 doubles)
    __shared__ double sOps[3 * MK];
    double* sAt = sOps;
    double* sAs = sOps + MK;
    double* sAr = sOps + 2 * MK;

    // Per-warp workspace: sB[K³] + tmpK[K²M] + tmpJ[KM²]
    __shared__ double ws[WARPS_PER_BLOCK_TC][K3 + KKM + KMM];
    double* sB   = ws[warpId];
    double* tmpK = ws[warpId] + K3;
    double* tmpJ = ws[warpId] + K3 + KKM;

    // ── Load operators cooperatively (ALL warps participate) ───────────
    {
        const int tid   = threadIdx.x;
        const int total = WARPS_PER_BLOCK_TC * 32;
        for (int i = tid; i < 3 * MK; i += total) {
            if (i < MK)
                sOps[i] = At[i];
            else if (i < 2 * MK)
                sOps[i] = As[i - MK];
            else
                sOps[i] = Ar[i - 2 * MK];
        }
    }

    // Load B slice into per-warp shared memory (before syncthreads)
    if (n < N) {
        for (int t = lane; t < K3; t += 32) {
            sB[t] = B[(size_t)n * LDB + t];
        }
    }

    __syncthreads();  // ensure operators visible; all warps participate

    if (n >= N) return;  // early exit AFTER syncthreads

    // ── Pre-load operator A-fragments into registers ───────────────────
    // MMA A is row-major 8×4: thread lane holds A[lane/4][lane%4]
    // Operators stored column-major: Op(m,k) = Op[k*M + m]
    // A[lane/4][lane%4] = Op[m=lane/4, k=lane%4] = Op[(lane%4)*M + lane/4]
    const double a_At = sAt[(lane % 4) * M + (lane / 4)];
    const double a_As = sAs[(lane % 4) * M + (lane / 4)];
    const double a_Ar = sAr[(lane % 4) * M + (lane / 4)];

    // ═══════════════════════════════════════════════════════════════════
    // Stage 1: tmpK[(m^j) + M*j + M*K*i] = sum_k At[m,k] * B[i + K*j + K²*k]
    //   MMA: At(8×4) × B_batch(4×8) → D(8×8), 2 batches of 8 (i,j) pairs
    //   XOR permutation on first index: physical = (m ^ j)
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int batch = 0; batch < NUM_BATCHES_S1; ++batch) {
        // B fragment: col-major 4×8, B_packed[k][n_col] = B[i + K*j + K²*k]
        // Thread lane: row = lane%4 = k, col = lane/4 = n_col
        const int n_col = lane / 4;          // 0..7
        const int k_idx = lane % 4;          // 0..3
        const int i_idx = 2 * batch + n_col / K;
        const int j_idx = n_col % K;
        const double b_val = sB[i_idx + K * j_idx + K * K * k_idx];

        double c0 = 0.0, c1 = 0.0;
        double d0, d1;

        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};"
            : "=d"(d0), "=d"(d1)
            : "d"(a_At), "d"(b_val), "d"(c0), "d"(c1)
        );

        // Result D[m][n]: m = lane/4, col0 = (lane%4)*2, col1 = col0+1
        const int m    = lane / 4;
        const int col0 = (lane % 4) * 2;
        const int col1 = col0 + 1;
        // Map output column back to (i, j)
        const int i0 = 2 * batch + col0 / K, j0 = col0 % K;
        const int i1 = 2 * batch + col1 / K, j1 = col1 % K;
        // XOR permutation: store at (m ^ j) instead of m
        tmpK[(m ^ j0) + M * j0 + M * K * i0] = d0;
        tmpK[(m ^ j1) + M * j1 + M * K * i1] = d1;
    }

    __syncwarp();

    // ═══════════════════════════════════════════════════════════════════
    // Stage 2: tmpJ[(c^b^ii) + M*b + M²*ii] = sum_j As[b,j] * tmpK[(c^j) + M*j + M*K*ii]
    //   MMA: As(8×4) × tmpK_i^T(4×8) → D(8×8), one per i value (K=4 calls)
    //   XOR permutation on tmpK read: physical = (c ^ j)
    //   XOR permutation on tmpJ write: physical = (c ^ b ^ ii)
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int ii = 0; ii < K; ++ii) {
        // B fragment: tmpK_i transposed → B[j][c] = tmpK[(c^j) + M*j + M*K*ii]
        // Thread lane: row = lane%4 = j, col = lane/4 = c
        const int c_rd = lane / 4;
        const int j_rd = lane % 4;
        const double b_val = tmpK[(c_rd ^ j_rd) + M * j_rd + M * K * ii];

        double c0 = 0.0, c1 = 0.0;
        double d0, d1;

        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};"
            : "=d"(d0), "=d"(d1)
            : "d"(a_As), "d"(b_val), "d"(c0), "d"(c1)
        );

        // D[b][c]: b = lane/4, c0 = (lane%4)*2, c1 = (lane%4)*2+1
        const int b_idx  = lane / 4;
        const int c0_idx = (lane % 4) * 2;
        const int c1_idx = c0_idx + 1;
        // XOR permutation: store at (c ^ b ^ ii) instead of c
        tmpJ[(c0_idx ^ b_idx ^ ii) + M * b_idx + M * M * ii] = d0;
        tmpJ[(c1_idx ^ b_idx ^ ii) + M * b_idx + M * M * ii] = d1;
    }

    __syncwarp();

    // ═══════════════════════════════════════════════════════════════════
    // Stage 3: C[n*LDC + a + M*b + M²*cc] = sum_i Ar[a,i] * tmpJ[(cc^b^i) + M*b + M²*i]
    //   MMA: Ar(8×4) × tmpJ_c^T(4×8) → D(8×8), one per c value (M=8 calls)
    //   XOR permutation on tmpJ read: physical = (cc ^ b ^ i)
    //   Original layout had 16-way bank conflicts; XOR reduces to 2-way.
    // ═══════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int cc = 0; cc < M; ++cc) {
        // B fragment: tmpJ_c transposed → B[i][b] = tmpJ[(cc^b^i) + M*b + M²*i]
        // Thread lane: row = lane%4 = i, col = lane/4 = b
        const int b_rd = lane / 4;
        const int i_rd = lane % 4;
        const double b_val = tmpJ[(cc ^ b_rd ^ i_rd) + M * b_rd + M * M * i_rd];

        double c0 = 0.0, c1 = 0.0;
        double d0, d1;

        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};"
            : "=d"(d0), "=d"(d1)
            : "d"(a_Ar), "d"(b_val), "d"(c0), "d"(c1)
        );

        // D[a][b]: a = lane/4, b0 = (lane%4)*2, b1 = (lane%4)*2+1
        const int a_idx  = lane / 4;
        const int b0_idx = (lane % 4) * 2;
        const int b1_idx = b0_idx + 1;
        C[(size_t)n * LDC + a_idx + M * b0_idx + M * M * cc] = d0;
        C[(size_t)n * LDC + a_idx + M * b1_idx + M * M * cc] = d1;
    }
}

// ── Dispatch wrapper ────────────────────────────────────────────────

void launchFusedTPKernel_TC(
    const double* d_Ar, const double* d_As, const double* d_At,
    const double* d_B, double* d_C,
    int M, int K, int N, int LDB, int LDC)
{
    if (M == 8 && K == 4) {
        constexpr int WPB = WARPS_PER_BLOCK_TC;
        const int blocks  = (N + WPB - 1) / WPB;
        const int threads = WPB * 32;
        fused_tp_tc_kernel<8, 4><<<blocks, threads>>>(
            d_Ar, d_As, d_At, d_B, d_C, N, LDB, LDC);
        return;
    }

    fprintf(stderr,
        "ERROR: fused_tp_tc_kernel has no instantiation for M=%d, K=%d\n"
        "Currently only M=8, K=4 is supported for Tensor Core path.\n",
        M, K);
}
