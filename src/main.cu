#include "common.cuh"
#include "../include/gemv.cuh"
#include <iostream>

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

    // TODO kernel gemv_naive.
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gemv_naive_kernel <<<grid_size, block_size>>>(d_A, d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_to_host(h_y, d_y, n);

    // TODO print.
std::cout << "y[0] = " << h_y[0] << std::endl;
std::cout << "y[1] = " << h_y[1] << std::endl;
    delete[] h_x;
    delete[] h_y;
    delete[] h_A;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}