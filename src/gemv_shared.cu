#include "../include/gemv.cuh"

/* gemv_shared_kernel.
One block computes one full dot product.
Step 1: Each thread accumulates a partial sum in shared memory,
        striding across the row by BLOCK_SIZE.
Step 2: Sums the block_size amount of partial sums in shared memory into one. 
Step 3: Thread 0 writes the final result to y.
*/

#define BLOCK_SIZE 256
__global__ void gemv_shared_kernel(const float* A, const float* x, float* y, size_t n) {
    size_t block_id = blockIdx.x;
    size_t thread_id = threadIdx.x;
    
    if (block_id < n) {
        __shared__ float sum[BLOCK_SIZE];

        sum[thread_id] = 0;
        __syncthreads();

        // Step 1.
        for (size_t j = thread_id; j < n; j+=BLOCK_SIZE) {
            sum[thread_id] += A[block_id * n + j] * x[j];
        }
        __syncthreads();

        // Step 2.
        for (size_t offset = BLOCK_SIZE/2; offset > 0; offset/=2) {
            if (thread_id < offset) {
                sum[thread_id] += sum[thread_id + offset];
            }
            __syncthreads();
        }

        // Step 3.
        if (thread_id == 0) {
            y[block_id] = sum[0];
        }
    }

}