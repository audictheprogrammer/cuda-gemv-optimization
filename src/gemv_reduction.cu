#include "../include/gemv.cuh"


/* gemv_reduction_kernel.
One warp (32 threads) computes one full dot product (one row of A times x).
Step 1: Each thread accumulates a partial sum in its own register (sum),
        striding across the row by 32.
Step 2: Sums the 32 partial sums into one. Uses __shfl_down_sync to read other thread's registers.
Step 3: Lane 0 holds the final result and writes in y[warp_id].
*/
__global__ void gemv_reduction_kernel(const float* A, const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t warp_id = i / 32;
    size_t lane = threadIdx.x % 32;
    if (warp_id < n) {
        float sum = 0;
        
        // Step 1.
        for (size_t j = lane; j < n; j+=32) {
            sum += A[warp_id * n + j] * x[j];
        }

        // Step 2.
        for (size_t offset=16; offset > 0; offset/=2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Step 3.
        if (lane == 0) {
            y[warp_id] = sum;
        } 

    }
}

