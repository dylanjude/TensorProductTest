#ifndef TENSOR_PRODUCT_CUDA_HPP
#define TENSOR_PRODUCT_CUDA_HPP

// Launch the fused tensor-product CUDA kernel.
// All pointers must be device pointers.
// Operators: Ar, As, At are M*K doubles each (row-major, m*K + k).
// B:  N slices of K*K*K doubles, stride LDB between slices.
// C:  N slices of M*M*M doubles, stride LDC between slices.
void launchFusedTPKernel(
    const double* d_Ar, const double* d_As, const double* d_At,
    const double* d_B, double* d_C,
    int M, int K, int N, int LDB, int LDC);

#endif // TENSOR_PRODUCT_CUDA_HPP
