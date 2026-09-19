#include "common.cuh"
#include <iostream>

int main () {
    const int n = 1024;

    double* h_A = alloc_host<double>(n*n);
    double* h_x = alloc_host<double>(n);
    double* h_y = alloc_host<double>(n);

    double* d_A = alloc_device<double>(n*n);
    double* d_x = alloc_device<double>(n);
    double* d_y = alloc_device<double>(n);

    fill<double>(h_A, n*n);
    fill<double>(h_x, n);

    copy_to_device<double>(d_A, h_A, n*n);
    copy_to_device<double>(d_x, h_x, n);

    // TODO kernel gemv_naive.

    copy_to_host<double>(h_y, d_y, n);

    // TODO print.

    delete[] h_x;
    delete[] h_y;
    delete[] h_A;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}