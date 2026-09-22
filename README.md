# cuda-gemv-optimization

Progressive optimization of a matrix-vector multiplication (GEMV) kernel in raw CUDA,
from a naive implementation to a warp-level reduction, with correctness verification
and memory bandwidth profiling using Nsight Compute.

## Why this project

During LLM inference, the decode phase generates one token at a time, often turning matrix-matrix multiplications (GEMM) into matrix-vector multiplications (GEMV) with batch size = 1.

GEMV is typically memory-bandwidth bound, making memory access patterns critical for performance.

This project explores this through a progressive optimization of a GEMV CUDA kernel, from a naive implementation to a warp-level reduction.

## Results

Matrix size: 1024×1024, float32, tested on NVIDIA Tesla T4.

| Kernel     | Time (ms) | DRAM throughput (% of peak) | Strategy |
|------------|-----------|------------------------------|----------|
| Naive      | 0.338     | 4.79%                        | 1 thread = 1 row, strided memory access |
| Coalesced  | 1.502     | 0.88%                        | 1 thread = 1 column, atomicAdd on y |
| Shared     | 0.123     | 44.68%                       | 1 block = 1 row, shared-memory reduction |
| Reduction  | 0.098     | 61.31%                       | 1 warp = 1 row, warp-level shuffle reduction |

Values are averaged over 5 runs on Google Colab (Tesla T4) to smooth out
run-to-run variance inherent to shared cloud infrastructure.
## Implementation walkthrough

### 1. Naive (`gemv_naive.cu`)
One thread computes one full dot product (one row of A × x).
Simple, but each thread in a warp reads memory addresses spaced `n` elements apart —
no coalescing, most of the theoretical memory bandwidth is left unused (4.79%).

### 2. Coalesced (`gemv_coalesced.cu`)
Flips the strategy: each thread handles one column instead of one row, so threads
in the same warp read contiguous memory — solving the coalescing problem.
But this creates a new bottleneck: many threads must now write to the same `y[j]`,
requiring `atomicAdd` and forcing serialized writes. The result is *worse* than
naive (0.88%) — proof that fixing one bottleneck can reveal another, and that
memory access pattern isn't the only thing that matters.

### 3. Shared-memory (`gemv_shared.cu`)
One block computes one full dot product.
Each thread processes a strided subset of the row and accumulates a partial sum in a register. The partial sums are then written to shared memory and combined using a tree reduction with `__syncthreads()`.

The reduction is much faster than both previous approaches, reaching 44.68% of peak DRAM throughput and reducing kernel time to 0.123 ms.

### 4. Warp-level reduction (`gemv_reduction.cu`)
Refines the shared-memory approach by assigning one warp (32 threads) to one row.
Each thread reads a strided subset of the row (coalesced access across the warp), accumulates a partial sum in a register, then all 32 partial sums are combined via `__shfl_down_sync`— a register-to-register exchange within the warp, with no shared memory and no global memory writes until the very end.
Only lane 0 writes the final result to `y[row]`, eliminating the atomic contention entirely.

Result: 61.31% of peak DRAM throughput, ~3.4x faster than naive and ~15.3x
faster than the atomicAdd version.

## Correctness verification

Each kernel's output is compared against a CPU reference implementation,
using relative error tolerance (not absolute) to account for floating-point
summation order differences between CPU and GPU — see `verify_gemv` in
`common.cuh`.

## Build & run

```bash
nvcc -arch=sm_XX src/main.cu src/gemv_naive.cu src/gemv_coalesced.cu src/gemv_shared.cu src/gemv_reduction.cu -o gemv_test
./gemv_test
```
Replace `sm_XX` with your GPU's compute capability (e.g. `sm_75` for T4, `sm_50` for
Maxwell-generation cards).

## Profiling with Nsight Compute

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./gemv_test
```

## What I'd explore next

- Larger matrix sizes and a roofline analysis (compute-bound vs memory-bound)
- Inline PTX for the critical load in the reduction kernel