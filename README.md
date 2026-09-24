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
| Naive      | 0.401     | 4.83%                        | 1 thread = 1 row, strided memory access |
| Coalesced  | 1.758     | 1.02%                        | 1 thread = 1 column, atomicAdd on y |
| Shared     | 0.134     | 44.90%                       | 1 block = 1 row, shared-memory reduction |
| Reduction  | 0.109     | 62.99%                       | 1 warp = 1 row, warp-level shuffle reduction |

Values are averaged over 5 runs on Google Colab (Tesla T4) to smooth out
run-to-run variance inherent to shared cloud infrastructure.

## Implementation walkthrough

### 1. Naive (`gemv_naive.cu`)
One thread computes one full dot product (one row of A × x).
Simple, but each thread in a warp reads memory addresses spaced `n` elements apart —
no coalescing, most of the theoretical memory bandwidth is left unused (4.83%).

### 2. Coalesced (`gemv_coalesced.cu`)
Flips the strategy: each thread handles one column instead of one row, so threads
in the same warp read contiguous memory — solving the coalescing problem.
But this creates a new bottleneck: many threads must now write to the same `y[j]`,
requiring `atomicAdd` and forcing serialized writes. The result is *worse* than
naive (1.02%) — proof that fixing one bottleneck can reveal another, and that
memory access pattern isn't the only thing that matters.

### 3. Shared-memory (`gemv_shared.cu`)
One block computes one full dot product.
Each thread processes a strided subset of the row and accumulates a partial sum in a register. The partial sums are then written to shared memory and combined using a tree reduction with `__syncthreads()`.

The reduction is much faster than both previous approaches, reaching 44.90% of peak DRAM throughput and reducing kernel time to 0.134 ms.

### 4. Warp-level reduction (`gemv_reduction.cu`)
Refines the shared-memory approach by assigning one warp (32 threads) to one row.
Each thread reads a strided subset of the row (coalesced access across the warp), accumulates a partial sum in a register, then all 32 partial sums are combined via `__shfl_down_sync`— a register-to-register exchange within the warp, with no shared memory and no global memory writes until the very end.
Only lane 0 writes the final result to `y[row]`, eliminating the atomic contention entirely.

Result: 62.99% of peak DRAM throughput, ~3.7x faster than naive and ~16.1x
faster than the atomicAdd version.

## Scaling behavior

| n     | Naive (ms) | Coalesced (ms) | Shared (ms) | Reduction (ms) |
|-------|------------|-----------------|-------------|-----------------|
| 1024  | 0.401      | 1.758           | 0.134       | 0.109           |
| 2048  | 0.567      | 2.646           | 0.076       | 0.075           |
| 4096  | 1.130      | 6.448           | 0.257       | 0.269           |
| 8192  | 4.202      | 15.716          | 1.009       | 1.025           |

### Why is n=1024 slower than n=2048, despite less work?

n=2048 has 4x more work than n=1024 (n² elements), yet Shared and Reduction
both run *faster* at n=2048. Why?

Both kernels have low occupancy at n=1024. When a warp stalls waiting for
memory, the SM switches to another ready warp, if one is available. With
fewer warps in reserve, SMs run out of ready work more often and sit idle.
At n=2048, more blocks are launched, more warps are in reserve, and stalls
get hidden more effectively (latency hiding). The extra idle time at n=1024
outweighs the extra work at n=2048.

A second, unverified factor could be the small number of blocks. With only 
128 blocks for Reduction at n=1024, the work may not be evenly  distributed 
across the GPU's SMs, and the total time may depend on the slowest one. 
Confirming this would need deeper profiling.

### Which kernel is actually better, Reduction or Shared?

| n     | Naive (%BW) | Coalesced (%BW) | Shared (%BW) | Reduction (%BW) |
|-------|-------------|------------------|--------------|-------------------|
| 1024  | 4.83        | 1.02             | 44.90        | 62.99             |
| 2048  | 9.35        | 1.98             | 76.48        | 80.75             |
| 4096  | 18.75       | 3.78             | 91.86        | 91.37             |
| 8192  | 21.58       | 5.93             | 96.22        | 96.48             |

| n     | Shared occupancy | Reduction occupancy |
|-------|-------------------|------------------------|
| 1024  | 89.6%             | 75.8%                  |
| 2048  | 92.4%             | 86.0%                  |
| 4096  | 95.5%             | 94.4%                  |
| 8192  | 97.8%             | 95.5%                  |

At n=1024, Reduction clearly wins on bandwidth (62.99% vs 44.90%) — two
things matter here.

The gap comes from their launch configs: Shared launches `<<<n, 256>>>`,
Reduction launches `<<<n/8, 256>>>` — one warp per row instead of one
block per row. This cuts two ways: Reduction gets fewer active warps
(lower occupancy: 75.8% vs 89.6%), but each thread does more work and
needs zero `__syncthreads()` calls, unlike Shared's fixed 8 rounds of
`__syncthreads()` per block (independent of n). At n=1024, skipping
synchronization outweighs the occupancy disadvantage — Reduction wins on
both bandwidth and time.

At large n, both effects fade: occupancy saturates for both kernels (see
table above), and Shared's fixed synchronization cost becomes a smaller
fraction of the growing per-thread workload. Bandwidth and occupancy
converge for both, and so does performance — both approach the GPU's
~96% memory bandwidth ceiling.

So: within the tested range (n=1024 to n=8192), both kernels converge to
the same memory bandwidth ceiling as n grows. The choice between them
seems to not matter. At small n, Reduction's lack of synchronization gives it
a real edge over Shared.

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
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ./gemv_test
```

## What I'd explore next

- A roofline analysis comparing this memory-bound GEMV against a compute-bound GEMM
- Inline PTX for the critical load in the reduction kernel
