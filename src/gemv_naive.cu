#include "../include/gemv.cuh"

/* gemv_naive_kernel.
One thread computes one full dot product.
*/
__global__ void gemv_naive_kernel(const float* A, const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 0;
        for (size_t j = 0; j < n; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }
}

