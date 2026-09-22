#include "common.cuh"
#include "../include/gemv.cuh"
#include <iostream>
#include <string>


void run_and_benchmark(const char* name,
                        void (*kernel)(const float*, const float*, float*, size_t),
                        const float* d_A, const float* d_x, float* d_y,
                        const float* h_A, const float* h_x, float* h_y,
                        size_t n, int grid_size, int block_size) {
    CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid_size, block_size>>>(d_A, d_x, d_y, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    copy_to_host(h_y, d_y, n);
    bool correct = verify_gemv(h_A, h_x, h_y, n);

    std::cout << name << " - time: " << milliseconds << " ms, "
              << (correct ? "correct" : "INCORRECT") << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main () {
    const int n = 1024;

    float* h_A = alloc_host(n*n);
    float* h_x = alloc_host(n);
    float* h_y = alloc_host(n);

    float* d_A = alloc_device(n*n);
    float* d_x = alloc_device(n);
    float* d_y = alloc_device(n);

    fill(h_A, n*n);
    fill(h_x, n);

    copy_to_device(d_A, h_A, n*n);
    copy_to_device(d_x, h_x, n);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // gemv_reduction_kernel assigns one warp (32 threads) per row, so we need
    // exactly n warps in total, rounded up to the nearest block.
    int warps_per_block = block_size / 32;
    int grid_size_reduction = (n + warps_per_block - 1) / warps_per_block;

    int grid_size_shared = n;

    run_and_benchmark("Naive",     gemv_naive_kernel,     d_A, d_x, d_y, h_A, h_x, h_y, n, grid_size,           block_size);
    run_and_benchmark("Coalesced", gemv_coalesced_kernel, d_A, d_x, d_y, h_A, h_x, h_y, n, grid_size,           block_size);
    run_and_benchmark("Shared",    gemv_shared_kernel,    d_A, d_x, d_y, h_A, h_x, h_y, n, grid_size_shared,    block_size);
    run_and_benchmark("Reduction", gemv_reduction_kernel, d_A, d_x, d_y, h_A, h_x, h_y, n, grid_size_reduction, block_size);

    delete[] h_x;
    delete[] h_y;
    delete[] h_A;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}