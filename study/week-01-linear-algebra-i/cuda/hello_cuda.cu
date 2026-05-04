// hello_cuda.cu
// Week 01 — Linear Algebra I | Lab Exercise
// Assad's 52-Week NVIDIA Curriculum
//
// First CUDA program: vector addition on GPU
//
// Demonstrates:
//   - cudaMalloc / cudaFree  (GPU heap — analogous to new/delete)
//   - cudaMemcpy Host→Device and Device→Host
//   - __global__ kernel declaration
//   - Thread indexing: threadIdx.x + blockIdx.x * blockDim.x
//   - Coalesced memory access: consecutive threads → consecutive addresses
//   - Error checking via cudaGetLastError()
//
// Compile: nvcc -o hello_cuda hello_cuda.cu
// Run:     ./hello_cuda
//
// Expected output:
//   All 1024 results should equal 2.000000 (1.0f + 1.0f)

#include <stdio.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// CUDA error checking macro
// Wrap every CUDA call with this in production code
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// ─────────────────────────────────────────────────────────────
// GPU Kernel: vectorAdd
//
// Each thread computes ONE element of the output:
//   c[i] = a[i] + b[i]
//
// Thread indexing:
//   threadIdx.x  → local ID within the block (0..blockDim.x-1)
//   blockIdx.x   → which block this thread is in
//   blockDim.x   → number of threads per block
//   idx = threadIdx.x + blockIdx.x * blockDim.x → global unique ID
//
// Memory access pattern:
//   Thread 0 → a[0], Thread 1 → a[1], Thread 2 → a[2] ...
//   Consecutive threads access consecutive addresses → COALESCED
//   This is the optimal GPU memory access pattern.
// ─────────────────────────────────────────────────────────────
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main() {
    printf("═══════════════════════════════════════════\n");
    printf("Week 01 — Hello CUDA: Vector Addition\n");
    printf("═══════════════════════════════════════════\n\n");

    // ── Parameters ───────────────────────────────────────────
    const int N          = 1024;
    const int THREADS    = 256;                          // threads per block
    const int BLOCKS     = (N + THREADS - 1) / THREADS; // blocks needed
    const size_t BYTES   = N * sizeof(float);

    printf("Vector size:    N = %d\n", N);
    printf("Threads/block:  %d\n", THREADS);
    printf("Blocks:         %d\n", BLOCKS);
    printf("Total threads:  %d\n\n", THREADS * BLOCKS);

    // ── Host arrays ──────────────────────────────────────────
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // Initialize: a[i] = 1.0, b[i] = 1.0 → expected c[i] = 2.0
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    // ── Device (GPU) arrays ──────────────────────────────────
    float *d_a, *d_b, *d_c;

    printf("Allocating GPU memory...\n");
    CUDA_CHECK(cudaMalloc(&d_a, BYTES));   // analogous to: new float[N] on GPU
    CUDA_CHECK(cudaMalloc(&d_b, BYTES));
    CUDA_CHECK(cudaMalloc(&d_c, BYTES));

    // ── Host → Device transfer ───────────────────────────────
    printf("Copying data Host → Device...\n");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));

    // ── Launch kernel ────────────────────────────────────────
    printf("Launching kernel: vectorAdd<<<%d, %d>>>...\n", BLOCKS, THREADS);
    vectorAdd<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Device → Host transfer ───────────────────────────────
    printf("Copying results Device → Host...\n\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost));

    // ── Verify results ───────────────────────────────────────
    printf("First 10 results (expected: 2.000000 each):\n");
    for (int i = 0; i < 10; ++i) {
        printf("  c[%3d] = %f%s\n", i, h_c[i],
               (h_c[i] == 2.0f) ? " ✓" : " ✗ WRONG");
    }

    // Full correctness check
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 2.0f) {
            errors++;
        }
    }
    printf("\nFull verification: %d / %d correct. Errors: %d\n",
           N - errors, N, errors);

    if (errors == 0) {
        printf("\n✓ MILESTONE ACHIEVED: First CUDA kernel ran correctly.\n");
        printf("  %d threads ran in parallel on the GPU.\n", N);
        printf("  Each computed c[i] = a[i] + b[i] independently.\n");
        printf("  Access pattern: coalesced (consecutive threads → consecutive addresses).\n");
    }

    // ── Print GPU info ───────────────────────────────────────
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("\n GPU: %s\n", prop.name);
    printf(" CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf(" Total Global Memory: %.1f GB\n",
           (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf(" SM Count: %d\n", prop.multiProcessorCount);
    printf(" Max Threads/Block: %d\n", prop.maxThreadsPerBlock);

    // ── Clean up ─────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_a));     // analogous to: delete[] on GPU
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    printf("\n═══════════════════════════════════════════\n");
    printf("Week 01 Lab Complete.\n");
    printf("Environment is ready for 51 more weeks.\n");
    printf("═══════════════════════════════════════════\n");

    return 0;
}
