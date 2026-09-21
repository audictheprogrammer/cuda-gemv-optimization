#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* Allocate dynamically a 1D-array of size n and returns it. */
inline float* alloc_host(size_t n) {
    return new float[n];
}

/* Fill the 1D array of size n by numbers from 0 to n-1. */
inline void fill(float* h_ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h_ptr[i] = i;
    }
}

/* CUDA allocates a 1D-array of size n and returns it. */
inline float* alloc_device(size_t n) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc((void**) &d_ptr, n * sizeof(float)));
    return d_ptr;
}

inline void copy_to_device(float* d_ptr, float* h_ptr, size_t n) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice));
}

inline void copy_to_host(float* h_ptr, float* d_ptr, size_t n) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
}

inline bool verify_gemv(const float* A, const float* x, const float* y_gpu, size_t n, float rel_tol = 1e-3f) {
    for (size_t i = 0; i < n; i++) {
        float expected = 0.0f;
        for (size_t j = 0; j < n; j++) {
            expected += A[i*n + j] * x[j];
        }
        float diff = fabs(expected - y_gpu[i]);
        float rel_error = diff / fmax(fabs(expected), 1e-6f);
        if (rel_error > rel_tol) {
            fprintf(stderr, "Mismatch at y[%zu]: expected %f, got %f (rel error %e)\n", i, expected, y_gpu[i], rel_error);
            return false;
        }
    }
    return true;
}