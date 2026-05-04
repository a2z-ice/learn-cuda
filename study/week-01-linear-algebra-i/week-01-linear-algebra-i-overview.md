# Week 1: Linear Algebra I — Overview & Study Plan
> Phase 1: Math & Systems | Assad's NVIDIA Curriculum | Week 1 of 52 | Milestone: Environment Ready

---

## 1. Learning Objectives

By end of Week 1, you must be able to do each of these — not just describe them, but *execute* them:

1. **Represent vectors and perform operations by hand and in code.** Given two vectors in ℝⁿ, compute their sum, scalar multiple, and dot product — both algebraically and in a raw C++ array implementation without any library.
2. **Explain vector spaces and basis vectors with precision.** Define a vector space, state its axioms from memory, and explain why the standard basis {e₁, e₂, e₃} spans ℝ³ — in plain English a NVIDIA engineer could evaluate in 90 seconds.
3. **Trace dot product to matrix multiplication to neural network inference.** Walk the path: dot product → weighted sum → single neuron → layer → forward pass. Identify exactly where GPU parallelism enters this chain.
4. **Write, compile, and run a C++ program using raw arrays and pointers.** Implement vector dot product in C++ using pointer arithmetic (no STL vectors), compile with `g++ -O2`, verify output, and explain stack vs heap allocation for your data.
5. **Stand up a functional GPU cloud environment.** Provision a RunPod instance, SSH in, verify CUDA toolkit (`nvcc --version`), and compile + run a starter CUDA kernel — with proof committed to GitHub.

---

## 2. Why This Matters for NVIDIA

### The Direct Line

Everything NVIDIA's GPU compute stack does — cuDNN convolutions, TensorRT inference, NeMo training — reduces to one operation at its mathematical core: the **dot product**. Understanding vectors, basis transformations, and linear operations at a rigorous level means you understand what the hardware is actually being asked to do before a single line of CUDA is written.

### Which Teams Own This

| NVIDIA Team | How Week 1 Math Connects |
|---|---|
| **cuDNN** | Every convolution is a series of dot products over input patches — understanding this is table stakes |
| **TensorRT** | Kernel fusion and layer optimization require knowing what linear algebra operations can be fused |
| **NeMo / LLM Training** | Attention is `softmax(QKᵀ/√d)` — three matrix operations built on dot products |
| **NCCL** | AllReduce aggregates gradients — knowing what a "gradient" is (a vector in weight space) is assumed |
| **CUDA Libraries** | cuBLAS is the BLAS primitive library; you will eventually write kernels that compete with it |

### What a NVIDIA Engineer Says

*"I don't care if you can recite the dot product formula. I want to know if you understand why two vectors being orthogonal means zero information transfer — and then I want to see that intuition in your GPU memory access patterns."*

Knowing this week's content deeply means you can speak to NVIDIA engineers as a peer who reasons about computation geometrically, not just syntactically.

---

## 3. Prerequisite Self-Check

Answer these three questions *right now* before you open a textbook. Check your answers below.

**Q1: What is the result of this dot product?**
`u = [2, -1, 3]`, `v = [4, 0, -2]`

> **Correct Answer:** `(2)(4) + (-1)(0) + (3)(-2) = 8 + 0 - 6 = 2`
> If you got this: proceed. If not: spend 30 minutes reviewing scalar multiplication and summation notation — Khan Academy "Vectors" unit is sufficient.

**Q2: What does it mean in C++ to declare `int arr[5]` vs `int* arr = new int[5]`?**

> **Correct Answer:** `int arr[5]` is allocated on the **stack** — fixed size, automatic lifetime, freed when the function returns. `new int[5]` allocates on the **heap** — dynamic size, manual lifetime, must be freed with `delete[]`. Stack is faster (no allocator overhead); heap is necessary for sizes unknown at compile time.
> If you are fuzzy on this: read the "Memory Layout" section of any C++ primer — 45 minutes maximum.

**Q3: In one sentence, what is an LLM doing when it generates the next token?**

> **Correct Answer:** It is computing a probability distribution over its vocabulary by running input token embeddings (vectors) through layers of weighted matrix operations, then sampling from that distribution.
> If you cannot say something close to this: that's fine — LLM Theory is this week's track. Just flag it as a gap.

---

## 4. Day-by-Day Study Plan (Monday – Friday)

Assume 2–3 focused hours per day. No marathon sessions — depth over duration.

### Monday — Math Foundation + C++ Environment
*Goal: Solid conceptual footing + working compiler*

| Task | Time |
|---|---|
| Read: Vectors and scalars — definition, geometric intuition, ℝⁿ notation | 30 min |
| Derive by hand: dot product formula from projection geometry (not just memorize) | 20 min |
| Install g++/clang++ locally or confirm RunPod instance is provisioned | 30 min |
| Write `dot_product.cpp`: implement dot product with raw `int[]` arrays and a `for` loop | 40 min |
| Test: verify against manual calculation from Q1 above | 20 min |
| Commit: `week-01-linear-algebra-i/dot_product.cpp` | 10 min |

### Tuesday — Vector Spaces + Basis Vectors
*Goal: Structural understanding of what a "space" means*

| Task | Time |
|---|---|
| Study: 8 axioms of a vector space — write them out, do not just read | 30 min |
| Work examples: show that ℝ² satisfies all 8 axioms for `u=[1,2]`, `v=[3,-1]` | 25 min |
| Study: linear independence, span, basis — standard basis {e₁, e₂, e₃} | 30 min |
| Write `basis_vectors.cpp`: represent e₁, e₂, e₃ as arrays; express `[3, -2, 5]` as linear combination; print it | 45 min |
| Write `week-01-notes.md`: your own plain-English explanation of why basis matters for ML | 20 min |

### Wednesday — Pointer Arithmetic + Stack vs Heap Deep Dive
*Goal: C++ memory model is not mysterious*

| Task | Time |
|---|---|
| Read: pointer declaration, dereferencing, pointer arithmetic (`ptr + 1` moves by `sizeof(type)`) | 30 min |
| Write `memory_demo.cpp`: allocate a float array on the heap, fill with values, iterate with pointer arithmetic (no `arr[i]` syntax), print, `delete[]` | 45 min |
| Trace: draw the stack frame and heap layout for your program on paper | 20 min |
| Refactor `dot_product.cpp`: rewrite using raw `float*` heap-allocated arrays, pointer arithmetic traversal | 30 min |
| Research: why does CUDA `cudaMalloc` allocate on GPU global memory, and how does it parallel `new` on CPU heap? Write 3 sentences in `week-01-notes.md` | 20 min |

### Thursday — LLM Theory: Complete Mental Map
*Goal: Tokens → weights → inference is a concrete, traceable process*

| Task | Time |
|---|---|
| Study: tokenization — byte-pair encoding, vocabulary, token IDs. Tokenize "Hello world" conceptually | 30 min |
| Study: embedding layer — each token ID maps to a vector in ℝᵈ (d typically 768, 4096, etc.) | 20 min |
| Study: parameters and weights — what is a "parameter"? How many does GPT-2 small have (117M)? Where do they live? | 25 min |
| Study: the forward pass at a high level — embedding → attention → FFN → logits → softmax → token | 30 min |
| Write `llm-mental-map.md`: draw the full pipeline as ASCII art or a table, annotating each stage with its mathematical operation and approximate tensor shape for GPT-2 small | 45 min |

### Friday — RunPod Setup + CUDA Hello World + GitHub Portfolio Polish
*Goal: Milestone achieved, documented, committed*

| Task | Time |
|---|---|
| Provision RunPod: 1x RTX 3080 or A4000 spot instance, PyTorch or CUDA 12.x template | 20 min |
| SSH in, verify: `nvcc --version`, `nvidia-smi`, `g++ --version` | 10 min |
| Write `hello_cuda.cu`: kernel that adds two vectors of 1024 floats; host code allocates with `cudaMalloc`, calls kernel with `<<<1, 1024>>>`, copies result back, prints first 5 values | 60 min |
| Compile: `nvcc -o hello_cuda hello_cuda.cu` — fix any errors | 20 min |
| Screenshot: terminal showing `nvcc --version` + program output. Save as `milestone-env-ready.png` | 10 min |
| Final commit: all files, clean README, full folder structure | 20 min |

---

## 5. GitHub Deliverable

**Folder:** `nvidia-curriculum/week-01-linear-algebra-i/`

```
week-01-linear-algebra-i/
├── README.md                              # Week overview, objectives, milestone status
├── week-01-linear-algebra-i-overview.md  # This full study plan document
├── week-01-notes.md                       # Your handwritten-style notes on vectors, basis, LLM mental map
├── llm-mental-map.md                      # Token → weights → inference pipeline with tensor shapes
├── cpp/
│   ├── dot_product.cpp                    # Dot product: raw int[] arrays, no STL
│   ├── basis_vectors.cpp                  # e1/e2/e3, linear combination representation
│   ├── memory_demo.cpp                    # Stack vs heap demo, pointer arithmetic traversal
│   └── Makefile                           # Compile all three with g++ -O2 -Wall
├── cuda/
│   └── hello_cuda.cu                      # Vector addition kernel, 1024 threads, verified output
└── milestone/
    ├── milestone-env-ready.png            # Screenshot: nvcc + nvidia-smi + program output
    └── milestone-env-ready.md            # Written validation note
```

### Exact Git Commit Message

```
feat(week-01): Linear Algebra I — math, C++ memory model, LLM mental map, CUDA env ready

- Implement dot product and basis vector representation in raw C++
- Demonstrate stack vs heap allocation with pointer arithmetic
- Document full LLM mental map: tokens → embeddings → weights → inference
- Provision RunPod GPU instance; compile and run first CUDA vector addition kernel
- Milestone achieved: environment ready (see milestone/milestone-env-ready.png)
```

### What Someone Sees When They Clone

A folder that demonstrates you can bridge math → C++ → GPU in Week 1. The `cpp/` files show systems depth. The `cuda/` file shows you didn't just read about GPUs — you ran code on one. The `llm-mental-map.md` shows you understand the destination. A NVIDIA recruiter or engineer skimming this repo sees a candidate who treats self-study like professional engineering work.

---

## 6. Success Criteria

Running code is not the bar. Understanding is the bar.

### Question 1 — From a cuDNN or cuBLAS engineer:

> *"Walk me through why the dot product is the computational primitive that justifies GPU architecture. Why does a GPU with 10,000 cores exist specifically because of this operation?"*

**Strong Answer:**

"The dot product of two length-N vectors requires N multiplications and N-1 additions — operations that are completely independent of each other. A CPU with 8 cores can only do 8 simultaneously. A GPU can schedule thousands of multiply-accumulate operations in parallel because they share no data dependencies. Matrix multiplication is just a structured collection of dot products — multiply a 1024×1024 matrix and you're computing over a million dot products. GPU architecture is essentially a machine purpose-built to make that parallel fan-out as fast as physics allows."

---

### Question 2 — From a NeMo / LLM training engineer:

> *"You said embedding vectors live in ℝᵈ. What does it mean geometrically when two token embeddings have a high dot product?"*

**Strong Answer:**

"A high dot product between two vectors means they point in roughly the same direction — they are geometrically similar. In embedding space, this is trained to mean semantic similarity. Tokens with high dot product embeddings tend to appear in similar contexts. This is exactly why attention in transformers uses QKᵀ — it's computing dot products between query and key vectors to score how much each token should attend to every other token. High score = similar direction = high relevance."

---

### Question 3 — From a systems/CUDA engineer:

> *"In your `memory_demo.cpp`, you allocated a float array on the heap and traversed it with pointer arithmetic. Why does that traversal pattern matter when you move to GPU programming?"*

**Strong Answer:**

"Sequential pointer traversal maps to coalesced memory access on a GPU. When consecutive threads access consecutive memory addresses — which is what pointer arithmetic `ptr + threadIdx.x` does — the hardware can serve all those requests in a single memory transaction rather than one per thread. This is called coalesced global memory access and it's one of the most important performance patterns in CUDA. Understanding that `ptr + 1` advances by `sizeof(float)` bytes on CPU translates directly to understanding why `arr[threadIdx.x]` is a good access pattern and `arr[threadIdx.x * stride]` can be catastrophic."

---

## 7. Milestone Validation — "Environment Ready"

### What It Actually Means

"Environment ready" is not "I have an account on RunPod." It means you have **demonstrated end-to-end GPU compute**: code written on your machine, compiled on a GPU instance, executed by actual CUDA cores, output verified. The environment is a tool you control — not a thing you have access to.

### Proof of Completion

You have hit this milestone when you can show **all four** of the following simultaneously in a single terminal screenshot:

1. `nvcc --version` → shows CUDA 12.x (or 11.x minimum)
2. `nvidia-smi` → shows your GPU model, driver version, memory
3. `./hello_cuda` → prints the first 5 values of your vector addition result (e.g., `result[0] = 2.000000`, confirming `1.0 + 1.0 = 2.0`)
4. The terminal hostname shows you are on a remote RunPod instance (not localhost)

### Starter CUDA Kernel for hello_cuda.cu

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Host arrays
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 1.0f; }

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<1, N>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("  result[%d] = %f\n", i, h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

```bash
# Compile and run
nvcc -o hello_cuda hello_cuda.cu
./hello_cuda
```

### How to Document It for GitHub Portfolio

```markdown
# Milestone: Environment Ready
**Date:** [your date]
**Instance:** RunPod — [GPU model, e.g. RTX 3080, 10GB VRAM]
**CUDA Version:** 12.x
**Status:** ✅ ACHIEVED

## Verification Steps
1. Provisioned RunPod spot instance with CUDA 12.x template
2. SSH access confirmed
3. `nvcc --version` → CUDA compilation tools, release 12.x
4. `nvidia-smi` → [GPU model] | Driver [version] | [VRAM] MiB
5. Compiled: `nvcc -o hello_cuda cuda/hello_cuda.cu`
6. Executed: `./hello_cuda` → correct vector addition output verified

## What I Now Have
- A GPU cloud environment I can spin up in <5 minutes
- A working CUDA compile/run workflow
- Baseline for all future weekly lab exercises
```

This note, combined with the screenshot, is the kind of artifact that tells a NVIDIA recruiter: this person doesn't write "familiar with cloud GPU environments" in a resume — they prove it.

---

*Assad's 52-Week NVIDIA Curriculum · Week 1: Linear Algebra I · Phase 1: Math & Systems*
*Saved: overview | File: week-01-linear-algebra-i-overview.md*
*Store at: `nvidia-curriculum/week-01-linear-algebra-i/week-01-linear-algebra-i-overview.md`*
