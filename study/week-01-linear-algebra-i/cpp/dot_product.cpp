// dot_product.cpp
// Week 01 — Linear Algebra I | CUDA / C++ Track
// Assad's 52-Week NVIDIA Curriculum
//
// Implements vector dot product using:
//   Version 1: raw int[] stack arrays with index-based loop
//   Version 2: heap-allocated float* arrays with pointer arithmetic
//
// Compile: g++ -O2 -Wall -o dot_product dot_product.cpp
// Run:     ./dot_product

#include <iostream>
#include <cstddef>   // size_t

// ─────────────────────────────────────────────────────────────
// Version 1: Stack arrays, index-based loop
// ─────────────────────────────────────────────────────────────
int dot_product_stack(int a[], int b[], int n) {
    int result = 0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ─────────────────────────────────────────────────────────────
// Version 2: Heap-allocated float*, pointer arithmetic traversal
// This traversal pattern maps directly to CUDA kernel indexing:
//   *(ptr + threadIdx.x) is how each GPU thread accesses its element
// ─────────────────────────────────────────────────────────────
float dot_product_heap(const float* a, const float* b, int n) {
    float result = 0.0f;
    const float* pa = a;   // pointer to first element of a
    const float* pb = b;   // pointer to first element of b
    for (int i = 0; i < n; ++i) {
        result += (*(pa + i)) * (*(pb + i));   // pointer arithmetic, no arr[i]
    }
    return result;
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "Week 01 — Dot Product Demo\n";
    std::cout << "═══════════════════════════════════════════\n\n";

    // ── Version 1: Stack arrays ──────────────────────────────
    int u[] = {2, -1, 3};
    int v[] = {4,  0, -2};
    int n   = 3;

    std::cout << "Version 1: Stack arrays, index loop\n";
    std::cout << "  u = [2, -1,  3]\n";
    std::cout << "  v = [4,  0, -2]\n";
    int result_int = dot_product_stack(u, v, n);
    std::cout << "  u · v = " << result_int << "\n";
    std::cout << "  Expected: (2)(4) + (-1)(0) + (3)(-2) = 8 + 0 - 6 = 2\n\n";

    // ── Version 2: Heap-allocated float* ─────────────────────
    int dim = 4;
    float* a = new float[dim];
    float* b = new float[dim];

    // Fill using pointer arithmetic (no a[i] syntax)
    *(a + 0) = 1.0f;  *(a + 1) = 2.0f;  *(a + 2) = 3.0f;  *(a + 3) = 4.0f;
    *(b + 0) = 4.0f;  *(b + 1) = 3.0f;  *(b + 2) = 2.0f;  *(b + 3) = 1.0f;

    std::cout << "Version 2: Heap float*, pointer arithmetic\n";
    std::cout << "  a = [1.0, 2.0, 3.0, 4.0]\n";
    std::cout << "  b = [4.0, 3.0, 2.0, 1.0]\n";
    float result_float = dot_product_heap(a, b, dim);
    std::cout << "  a · b = " << result_float << "\n";
    std::cout << "  Expected: 1*4 + 2*3 + 3*2 + 4*1 = 4+6+6+4 = 20.0\n\n";

    // ── Memory: print addresses to see heap placement ─────────
    std::cout << "Memory layout (heap):\n";
    for (int i = 0; i < dim; ++i) {
        std::cout << "  a[" << i << "] @ address " << (a + i)
                  << "  value = " << *(a + i) << "\n";
    }
    std::cout << "  Address gap between elements: "
              << sizeof(float) << " bytes (sizeof float)\n\n";

    // ── Orthogonality demo ───────────────────────────────────
    float orth_a[] = {1.0f, 0.0f};
    float orth_b[] = {0.0f, 1.0f};
    float ortho_result = dot_product_heap(orth_a, orth_b, 2);
    std::cout << "Orthogonality check:\n";
    std::cout << "  e1 = [1, 0], e2 = [0, 1]\n";
    std::cout << "  e1 · e2 = " << ortho_result
              << "  (0 → orthogonal → no shared direction)\n\n";

    // ── Clean up heap ─────────────────────────────────────────
    delete[] a;
    delete[] b;

    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "GPU connection: Each multiply-accumulate in this loop\n";
    std::cout << "is independent — a GPU runs all N of them in parallel.\n";
    std::cout << "That is why GPU architecture exists.\n";
    std::cout << "═══════════════════════════════════════════\n";

    return 0;
}
