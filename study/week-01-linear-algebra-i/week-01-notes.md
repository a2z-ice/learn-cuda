# Week 01 — Study Notes
> Linear Algebra I | Phase 1: Math & Systems | Assad's NVIDIA Curriculum

---

## Track 1: Mathematics — Vectors & Linear Algebra

### What Is a Vector?

A **vector** is an ordered list of numbers that represents a point or direction in space.

- In ℝ²: `v = [3, -2]` — a point in 2D space
- In ℝ³: `v = [1, 4, -7]` — a point in 3D space
- In ℝⁿ: `v = [v₁, v₂, ..., vₙ]` — a point in n-dimensional space

In ML/DL, almost everything is a vector in very high-dimensional space:
- A token embedding might live in ℝ⁴⁰⁹⁶
- A neural network weight row is a vector
- A gradient is a vector pointing in the direction of steepest loss increase

### Scalars vs Vectors

| Concept | Definition | Example |
|---|---|---|
| **Scalar** | A single number | `λ = 3.14` |
| **Vector** | Ordered list of numbers | `v = [2, -1, 5]` |
| **Scalar × Vector** | Scale every component | `3 * [1, 2] = [3, 6]` |

### The Dot Product

Given `u = [u₁, u₂, ..., uₙ]` and `v = [v₁, v₂, ..., vₙ]`:

```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ  =  Σᵢ uᵢvᵢ
```

**Example:**
```
u = [2, -1, 3]
v = [4,  0, -2]

u · v = (2)(4) + (-1)(0) + (3)(-2)
      = 8 + 0 - 6
      = 2
```

**Geometric meaning:**
```
u · v = ||u|| ||v|| cos(θ)
```
- `θ = 0°`  → vectors point same direction → dot product = ||u||·||v|| (maximum)
- `θ = 90°` → vectors are orthogonal → dot product = 0 (no similarity)
- `θ = 180°`→ vectors point opposite → dot product = -||u||·||v|| (minimum)

**Why this matters for NVIDIA:**
Dot product is the primitive that matrix multiplication is built from, which is the primitive that neural network layers are built from, which is the primitive that attention is built from. GPU architecture exists largely to accelerate this single operation at massive scale.

---

### Vector Spaces

A **vector space** V is a set of vectors with two operations (addition, scalar multiplication) that satisfy 8 axioms:

| # | Axiom | Plain English |
|---|---|---|
| 1 | u + v ∈ V | Adding two vectors stays in the space |
| 2 | u + v = v + u | Addition is commutative |
| 3 | (u+v)+w = u+(v+w) | Addition is associative |
| 4 | ∃ 0 ∈ V: v + 0 = v | A zero vector exists |
| 5 | ∃ -v: v + (-v) = 0 | Every vector has an additive inverse |
| 6 | λv ∈ V | Scaling stays in the space |
| 7 | λ(u+v) = λu + λv | Distributive over vector addition |
| 8 | (λ+μ)v = λv + μv | Distributive over scalar addition |

ℝⁿ satisfies all 8 axioms — it is a vector space.

**Verification example for ℝ² with u=[1,2], v=[3,-1]:**
- Axiom 1: [1,2] + [3,-1] = [4,1] ∈ ℝ² ✓
- Axiom 6: 3 * [1,2] = [3,6] ∈ ℝ² ✓

---

### Basis Vectors

A **basis** for a vector space V is a set of vectors that:
1. **Spans** V — every vector in V can be written as a linear combination of the basis vectors
2. Is **linearly independent** — no basis vector can be written as a combination of the others

**Standard basis for ℝ³:**
```
e₁ = [1, 0, 0]
e₂ = [0, 1, 0]
e₃ = [0, 0, 1]
```

Any vector in ℝ³ can be decomposed:
```
[3, -2, 5] = 3·e₁ + (-2)·e₂ + 5·e₃
           = 3·[1,0,0] + (-2)·[0,1,0] + 5·[0,0,1]
```

**Why basis matters for ML:**
When we say a token embedding is "in ℝ⁴⁰⁹⁶," we mean the embedding space has 4096 basis dimensions. The model learns which combinations of these dimensions encode semantic meaning. Attention computes similarity between vectors in this space using — dot products.

---

## Track 2: CUDA / C++ — Memory Model

### Pointers Fundamentals

A **pointer** is a variable that holds a memory address.

```cpp
int x = 42;
int* ptr = &x;    // ptr holds the address of x
int val = *ptr;   // dereference: val = 42
```

### Stack vs Heap

| | Stack | Heap |
|---|---|---|
| **Allocation** | Automatic (variable declaration) | Manual (`new` / `malloc`) |
| **Deallocation** | Automatic (function returns) | Manual (`delete` / `free`) |
| **Size** | Fixed at compile time | Dynamic, runtime-determined |
| **Speed** | Faster (no allocator) | Slower (allocator overhead) |
| **Failure** | Stack overflow | `std::bad_alloc` |

```cpp
// Stack allocation
int arr[5] = {1, 2, 3, 4, 5};   // lives on stack, auto-freed

// Heap allocation
int* arr = new int[5];            // lives on heap
// ... use arr ...
delete[] arr;                     // must free manually — never forget this
```

### Pointer Arithmetic

When you add 1 to a pointer, it advances by `sizeof(type)` bytes:

```cpp
float* ptr = arr;    // points to arr[0]
ptr + 1;             // points to arr[1] (advances 4 bytes for float)
ptr + 2;             // points to arr[2] (advances 8 bytes)
*(ptr + i)           // same as arr[i]
```

This is critical for GPU programming:
- CUDA kernels use `base_ptr + threadIdx.x` to give each thread its own element
- Coalesced memory access (consecutive threads → consecutive addresses) is the fastest GPU access pattern
- Understanding pointer arithmetic makes CUDA memory indexing intuitive, not magical

### CUDA Memory Analogy

```
CPU (Host)                          GPU (Device)
──────────────────────────────      ──────────────────────────────
int* arr = new int[N];         ←→   cudaMalloc(&d_arr, N*sizeof(int));
// heap allocation                  // GPU global memory allocation

memcpy(dst, src, bytes);       ←→   cudaMemcpy(d_arr, h_arr, bytes, H2D);
// CPU memory copy                  // Host → Device transfer

delete[] arr;                  ←→   cudaFree(d_arr);
// heap deallocation                // GPU memory deallocation
```

The mental model is the same — you are managing a memory resource manually. The difference is which processor owns the memory and which bus moves the data.

---

## Track 3: LLM Theory — Complete Mental Map

### What Is an LLM?

A **Large Language Model** is a neural network trained to predict the next token given a sequence of previous tokens. "Large" refers to the number of parameters (weights) — modern LLMs have billions.

### Step 1: Tokenization

Text is not fed to the model as characters or words — it is broken into **tokens**.

```
"Hello, world!" → ["Hello", ",", " world", "!"] → [15496, 11, 995, 0]
```

- Tokens are subword units produced by Byte-Pair Encoding (BPE)
- Each token maps to an integer ID from the vocabulary
- Vocabulary size is typically 32k–128k tokens

### Step 2: Embedding Layer

Each token ID is looked up in an **embedding matrix** (shape: `[vocab_size, d_model]`):

```
Token ID 15496 → row 15496 of embedding matrix → vector in ℝᵈ_model
```

- For GPT-2 small: `d_model = 768`, so each token becomes a vector in ℝ⁷⁶⁸
- This is a learned lookup table — the model learns what each token "means" geometrically

### Step 3: Transformer Layers (Attention + FFN)

Each layer applies two sub-operations:

**Multi-Head Self-Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```
- Q, K, V are linear projections of the input embeddings
- `QKᵀ` is a matrix of dot products — measuring similarity between all pairs of tokens
- Output: a new representation for each token that "knows" about other tokens

**Feed-Forward Network (FFN):**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
- Two linear transformations with a ReLU activation
- Applied independently to each token position

### Step 4: Logits → Softmax → Token

```
Final hidden state → Linear(d_model, vocab_size) → logits [shape: vocab_size]
                  → softmax → probability distribution over vocabulary
                  → sample or argmax → next token ID
```

### GPT-2 Small — Tensor Shapes at Each Stage

| Stage | Operation | Tensor Shape |
|---|---|---|
| Input | Token IDs | `[seq_len]` |
| Embedding | Lookup | `[seq_len, 768]` |
| Attention QKV | Linear projection | `[seq_len, 768]` each |
| Attention scores | `QKᵀ / √64` | `[seq_len, seq_len]` per head |
| Attention output | Weighted sum of V | `[seq_len, 768]` |
| FFN hidden | `xW₁` → ReLU | `[seq_len, 3072]` |
| FFN output | `→ W₂` | `[seq_len, 768]` |
| Logits | Final linear | `[seq_len, 50257]` |
| Next token | argmax/sample | scalar |

GPT-2 small has 12 layers. The above repeats 12 times.
Total parameters: ~117 million.

### What Are Parameters?

**Parameters** (also called **weights**) are the numbers the model learned during training. They live in the weight matrices:
- Embedding matrix: `vocab_size × d_model`
- Each attention layer: Q, K, V, O projection matrices
- Each FFN layer: W₁, W₂ matrices
- All stored as float32 or float16 tensors in GPU memory

---

## Track 4: Lab Notes

### RunPod Setup Checklist

- [ ] Create RunPod account at runpod.io
- [ ] Launch spot instance: RTX 3080 or A4000 (cheapest with 10GB+ VRAM)
- [ ] Select template: **RunPod PyTorch** or **CUDA 12.x**
- [ ] Connect via SSH: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa`
- [ ] Verify: `nvcc --version` and `nvidia-smi`

### Environment Verification Commands

```bash
# CUDA compiler version
nvcc --version

# GPU hardware status
nvidia-smi

# C++ compiler
g++ --version

# Compile CUDA program
nvcc -o hello_cuda cuda/hello_cuda.cu

# Run
./hello_cuda
```

### Key Takeaway

The CUDA programming model mirrors C++ heap allocation. Understanding `new`/`delete` and pointer arithmetic on CPU makes `cudaMalloc`/`cudaFree` and kernel indexing (`ptr + threadIdx.x`) immediately intuitive. This is why C++ memory fundamentals are Week 1 — they are the prerequisite for everything in CUDA.

---

*Assad's 52-Week NVIDIA Curriculum · Week 1: Linear Algebra I*
