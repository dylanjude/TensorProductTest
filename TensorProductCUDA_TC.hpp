#ifndef TENSOR_PRODUCT_CUDA_TC_HPP
#define TENSOR_PRODUCT_CUDA_TC_HPP

// Launch the fused tensor-product kernel using FP64 Tensor Cores.
// Uses mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 PTX instruction.
// Requires SM >= 8.0 (Ampere or newer).
// Supports M <= 8, K <= 8 (common pairs pre-instantiated via dispatch table).
// All pointers must be device pointers.
void launchFusedTPKernel_TC(
    const double* d_Ar, const double* d_As, const double* d_At,
    const double* d_B, double* d_C,
    int M, int K, int N, int LDB, int LDC);

#endif // TENSOR_PRODUCT_CUDA_TC_HPP
