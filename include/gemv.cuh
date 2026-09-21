#pragma once
#include <cuda_runtime.h>

__global__ void gemv_naive_kernel(const float* A, const float* x, float* y, size_t n);
__global__ void gemv_coalesced_kernel(const float* A, const float* x, float* y, size_t n);
__global__ void gemv_reduction_kernel(const float* A, const float* x, float* y, size_t n);

