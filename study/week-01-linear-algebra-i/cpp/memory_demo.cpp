// memory_demo.cpp
// Week 01 — Linear Algebra I | CUDA / C++ Track
// Assad's 52-Week NVIDIA Curriculum
//
// Demonstrates:
//   - Stack vs heap allocation
//   - Pointer arithmetic (no arr[i] syntax in traversal)
//   - Memory address inspection
//   - Why this directly maps to CUDA global memory access patterns
//
// Compile: g++ -O2 -Wall -o memory_demo memory_demo.cpp
// Run:     ./memory_demo

#include <iostream>
#include <cstddef>   // size_t, ptrdiff_t

// ─────────────────────────────────────────────────────────────
// Print array using ONLY pointer arithmetic (no arr[i])
// ─────────────────────────────────────────────────────────────
void print_array_ptr(const char* label, const float* base_ptr, int n) {
    std::cout << "  " << label << ": [";
    for (int i = 0; i < n; ++i) {
        std::cout << *(base_ptr + i);
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ─────────────────────────────────────────────────────────────
// Fill array using only pointer dereferencing
// ─────────────────────────────────────────────────────────────
void fill_sequential(float* base_ptr, int n) {
    for (int i = 0; i < n; ++i) {
        *(base_ptr + i) = (float)(i + 1);   // 1.0, 2.0, 3.0, ...
    }
}

// ─────────────────────────────────────────────────────────────
// Dot product — pointer arithmetic, heap arrays
// Returns: sum of (*(a+i)) * (*(b+i))
// ─────────────────────────────────────────────────────────────
float dot_ptr(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += (*(a + i)) * (*(b + i));
    }
    return acc;
}

// ─────────────────────────────────────────────────────────────
// Stack allocation demo — fixed size, automatic lifetime
// ─────────────────────────────────────────────────────────────
void demo_stack() {
    std::cout << "── STACK ALLOCATION ────────────────────────────\n";
    float stack_arr[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    std::cout << "  float stack_arr[5] declared inside function\n";
    std::cout << "  Lives on the stack — auto-freed when function returns\n";
    std::cout << "  Cannot resize at runtime\n\n";

    std::cout << "  Addresses (sequential, 4 bytes apart for float):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "    stack_arr[" << i << "] @ " << (stack_arr + i)
                  << "  value = " << *(stack_arr + i) << "\n";
    }

    // Show byte distance between consecutive elements
    ptrdiff_t gap = (stack_arr + 1) - stack_arr;   // in units of float
    std::cout << "\n  Pointer gap between elements: " << gap
              << " float-unit(s) = " << gap * sizeof(float) << " bytes\n";
    std::cout << "  (sizeof(float) = " << sizeof(float) << " bytes)\n\n";
}

// ─────────────────────────────────────────────────────────────
// Heap allocation demo — dynamic size, manual lifetime
// ─────────────────────────────────────────────────────────────
void demo_heap(int n) {
    std::cout << "── HEAP ALLOCATION ─────────────────────────────\n";
    std::cout << "  n = " << n << " (determined at runtime)\n";
    std::cout << "  float* heap_arr = new float[n];\n\n";

    float* heap_arr = new float[n];   // heap allocation

    // Fill without arr[i]
    fill_sequential(heap_arr, n);
    print_array_ptr("heap_arr", heap_arr, n);

    std::cout << "\n  Addresses (heap — not guaranteed contiguous with stack):\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "    *(heap_arr + " << i << ") @ " << (heap_arr + i)
                  << "  value = " << *(heap_arr + i) << "\n";
    }

    std::cout << "\n  ⚠ Must free manually:\n";
    delete[] heap_arr;   // MUST free to avoid memory leak
    heap_arr = nullptr;  // good practice — dangling pointer prevention
    std::cout << "  delete[] heap_arr; executed\n";
    std::cout << "  heap_arr set to nullptr\n\n";
}

// ─────────────────────────────────────────────────────────────
// Show why pointer arithmetic matters for CUDA
// ─────────────────────────────────────────────────────────────
void demo_cuda_analogy(int n) {
    std::cout << "── CUDA MEMORY ANALOGY ─────────────────────────\n";

    float* a = new float[n];
    float* b = new float[n];

    // Fill a = [1,2,...,n], b = [n,n-1,...,1]
    for (int i = 0; i < n; ++i) {
        *(a + i) = (float)(i + 1);
        *(b + i) = (float)(n - i);
    }

    std::cout << "\n  CPU: dot_ptr(a, b, " << n << ")\n";
    std::cout << "  Each iteration: acc += (*(a+i)) * (*(b+i))\n";
    std::cout << "  → Sequential. One core. One multiply-accumulate at a time.\n\n";

    float result = dot_ptr(a, b, n);
    std::cout << "  Result = " << result << "\n\n";

    std::cout << "  GPU equivalent (conceptual):\n";
    std::cout << "  __global__ void dot_kernel(float* a, float* b, float* out) {\n";
    std::cout << "      int i = threadIdx.x;          // each thread gets unique i\n";
    std::cout << "      out[i] = *(a+i) * *(b+i);    // ALL done in parallel\n";
    std::cout << "  }\n";
    std::cout << "  → " << n << " threads. All " << n << " multiply-accumulates at once.\n";
    std::cout << "  → This is exactly why GPU architecture exists.\n\n";

    std::cout << "  CUDA vs C++ memory:\n";
    std::cout << "    new float[n]       ←→  cudaMalloc(&d_ptr, n*sizeof(float))\n";
    std::cout << "    delete[] ptr       ←→  cudaFree(d_ptr)\n";
    std::cout << "    memcpy(dst,src,b)  ←→  cudaMemcpy(d,h,b,cudaMemcpyHostToDevice)\n";
    std::cout << "    *(ptr + i)         ←→  *(d_ptr + threadIdx.x)\n\n";

    delete[] a;
    delete[] b;
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "Week 01 — Memory Model & Pointer Arithmetic\n";
    std::cout << "═══════════════════════════════════════════\n\n";

    demo_stack();
    demo_heap(6);
    demo_cuda_analogy(8);

    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "Key Rule: Every new[] needs a delete[]\n";
    std::cout << "          Every cudaMalloc needs a cudaFree\n";
    std::cout << "          Same discipline. Different memory space.\n";
    std::cout << "═══════════════════════════════════════════\n";

    return 0;
}
