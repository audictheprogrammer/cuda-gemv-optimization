#include "../include/gemv.cuh"

/* gemv_coalesced_kernel.
Each thread handles one column of A instead of one row.
This makes memory access coalasced (threads in a warp accessing contiguous data at the same time).
However, it adds the cost of making atomic add on y, since all the threads now write to the same y[j].
*/
__global__ void gemv_coalesced_kernel(const float* A, const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (size_t j = 0; j < n; j++) {
            atomicAdd(&y[j], A[j*n + i] * x[i]);
        }
    }
}

