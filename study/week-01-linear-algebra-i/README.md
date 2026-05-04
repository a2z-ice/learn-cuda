# Week 01 — Linear Algebra I
> **Phase 1: Math & Systems** | Assad's 52-Week NVIDIA Curriculum | Week 1 of 52

---

## Milestone Status

| Milestone | Status |
|---|---|
| Environment ready (RunPod + CUDA) | ✅ Target for Week 1 |
| dot_product.cpp compiles and runs | ✅ |
| basis_vectors.cpp compiles and runs | ✅ |
| memory_demo.cpp compiles and runs | ✅ |
| hello_cuda.cu compiles and runs on GPU | ✅ |
| LLM mental map documented | ✅ |

---

## This Week's Four Tracks

| Track | Topic | Key Deliverable |
|---|---|---|
| Mathematics | Vectors, dot product, vector spaces, basis | `week-01-notes.md` |
| CUDA / C++ | Pointers, raw arrays, stack vs heap | `cpp/` folder |
| LLM Theory | Tokens, weights, parameters, forward pass | `llm-mental-map.md` |
| Lab | RunPod setup + first CUDA kernel | `cuda/hello_cuda.cu` + `milestone/` |

---

## Learning Objectives

1. Compute dot products algebraically and in raw C++ without STL
2. Explain vector spaces and basis vectors precisely
3. Trace dot product → matrix multiply → neural network inference
4. Implement C++ programs using pointer arithmetic and heap allocation
5. Provision a GPU cloud environment and run a CUDA kernel end-to-end

---

## Folder Structure

```
week-01-linear-algebra-i/
├── README.md                              ← You are here
├── week-01-linear-algebra-i-overview.md  ← Full study plan & NVIDIA interview prep
├── week-01-notes.md                       ← Personal notes: vectors, basis, LLM mental map
├── llm-mental-map.md                      ← Token → weights → inference pipeline
├── cpp/
│   ├── dot_product.cpp                    ← Dot product, raw int[] arrays, no STL
│   ├── basis_vectors.cpp                  ← Standard basis e1/e2/e3, linear combination
│   ├── memory_demo.cpp                    ← Stack vs heap, pointer arithmetic traversal
│   └── Makefile                           ← Compiles all three with g++ -O2 -Wall
├── cuda/
│   └── hello_cuda.cu                      ← Vector addition kernel, 1024 threads
└── milestone/
    ├── milestone-env-ready.md             ← Written validation of environment milestone
    └── milestone-env-ready.png            ← [Screenshot: nvcc + nvidia-smi + output]
```

---

## How to Build and Run

### C++ Programs

```bash
cd cpp/
make            # builds all three binaries
./dot_product
./basis_vectors
./memory_demo
```

### CUDA Program (requires GPU + NVCC)

```bash
cd cuda/
nvcc -o hello_cuda hello_cuda.cu
./hello_cuda
```

Expected output:
```
First 5 results:
  result[0] = 2.000000
  result[1] = 2.000000
  result[2] = 2.000000
  result[3] = 2.000000
  result[4] = 2.000000
```

---

## Git Commit for This Week

```
feat(week-01): Linear Algebra I — math, C++ memory model, LLM mental map, CUDA env ready

- Implement dot product and basis vector representation in raw C++
- Demonstrate stack vs heap allocation with pointer arithmetic
- Document full LLM mental map: tokens → embeddings → weights → inference
- Provision RunPod GPU instance; compile and run first CUDA vector addition kernel
- Milestone achieved: environment ready (see milestone/milestone-env-ready.md)
```

---

*Assad's 52-Week NVIDIA Curriculum · Week 1 of 52 · Phase 1: Math & Systems*
