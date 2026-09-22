#include "../include/gemv.cuh"

/* gemv_shared_kernel.
One block computes one full dot product.
Step 1: Each thread accumulates a partial sum locally,
        striding across the row by BLOCK_SIZE.
Step 2: Each thread writes its partial sum into shared memory.
Step 3: Sums the block_size amount of partial sums in shared memory into one. 
Step 4: Thread 0 writes the final result to y.
*/

#define BLOCK_SIZE 256
__global__ void gemv_shared_kernel(const float* A, const float* x, float* y, size_t n) {
    size_t block_id = blockIdx.x;
    size_t thread_id = threadIdx.x;
    __shared__ float shared_mem[BLOCK_SIZE];

    if (block_id < n) {
        float sum = 0;

        // Step 1.
        for (size_t j = thread_id; j < n; j+=BLOCK_SIZE) {
            sum += A[block_id * n + j] * x[j];
        }

        // Step 2.
        shared_mem[thread_id] = sum;
        __syncthreads();

        // Step 3.
        for (size_t offset = BLOCK_SIZE/2; offset > 0; offset/=2) {
            if (thread_id < offset) {
                shared_mem[thread_id] += shared_mem[thread_id + offset];
            }
            __syncthreads();
        }

        // Step 4.
        if (thread_id == 0) {
            y[block_id] = shared_mem[0];
        }
    }

}