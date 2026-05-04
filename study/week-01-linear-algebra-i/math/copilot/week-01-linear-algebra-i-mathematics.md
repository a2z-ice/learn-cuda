# Week 1: Linear Algebra I — Mathematics
> Phase 1: Math & Systems | Assad's NVIDIA Curriculum | Week 1 of 52
> Topic: Vectors, scalars, dot product, vector spaces, basis vectors

---

## 1. Intuition Before Symbols

### Why Linear Algebra for GPUs?

Every pixel rendered, every neural network weight updated, every physics simulation stepped forward — all of these reduce to **operations on arrays of numbers**. A GPU is, at its core, a machine that performs thousands of these operations **simultaneously**.

Linear algebra gives us the language to describe those operations precisely:

- A **scalar** is a single number — a brightness value, a weight, a temperature.
- A **vector** is an ordered list of scalars — a pixel's RGB color `(255, 128, 0)`, a point in 3D space `(x, y, z)`, or a row of 4096 neural network weights.
- The **dot product** compresses two vectors into a single scalar — this one operation is the heartbeat of matrix multiplication, which is the heartbeat of deep learning.

### Think Physically

Imagine you're standing in a room. You can describe your position with three numbers: how far right, how far forward, and how far up. Those three directions are your **basis vectors**. Any position in the room is a **linear combination** of those three directions. Change the basis, and the same physical point gets different numbers — but the point itself doesn't move.

This is the core idea of a **vector space**: a set of objects (vectors) that you can add together and scale, governed by consistent rules.

### The GPU Connection

When a GPU runs a CUDA kernel, each thread typically processes one element (or a small tile) of a vector or matrix. Understanding vectors isn't abstract math — it's understanding **what each thread is computing**.

---

## 2. Formal Definitions

### 2.1 Scalar

A **scalar** is an element of a field (typically ℝ, the real numbers, or ℝ in floating-point approximation on hardware).

> **Notation:** Scalars are denoted by lowercase italic letters: *a*, *b*, *c*, *α*, *β*

A scalar has:
- **Magnitude** (absolute value)
- **No direction**

On a GPU, a scalar maps to a single register value — `float`, `double`, `half`, or `int`.

### 2.2 Vector

A **vector** in ℝⁿ is an ordered n-tuple of real numbers:

> **v** = (v₁, v₂, ..., vₙ) ∈ ℝⁿ

**Notation conventions:**
- Boldface lowercase: **v**, **u**, **w**
- Arrow notation: v⃗
- Column notation (standard in linear algebra):

```
    ┌ v₁ ┐
v = │ v₂ │
    │ .  │
    └ vₙ ┘
```

**Key properties of vectors:**
- **Dimension:** The number of components (n)
- **Magnitude (norm):** The "length" of the vector
- **Direction:** Where the vector "points" in n-dimensional space

### 2.3 Vector Addition

For **u** = (u₁, u₂, ..., uₙ) and **v** = (v₁, v₂, ..., vₙ):

> **u** + **v** = (u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ)

Addition is performed **component-wise**. On a GPU, this means each thread can independently add its assigned component — **perfect parallelism**.

### 2.4 Scalar Multiplication

For scalar *c* and vector **v** = (v₁, v₂, ..., vₙ):

> *c* **v** = (c·v₁, c·v₂, ..., c·vₙ)

Each component is scaled independently. Again, **embarrassingly parallel**.

### 2.5 Dot Product (Inner Product)

For **u** = (u₁, u₂, ..., uₙ) and **v** = (v₁, v₂, ..., vₙ):

> **u** · **v** = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σᵢ uᵢvᵢ

The dot product takes two vectors and returns a **scalar**. This is a **reduction operation** — it collapses n multiplications into a single sum. On GPUs, this requires **cooperative computation** across threads (unlike addition, which is purely independent).

### 2.6 Vector Space

A **vector space** V over a field F is a set equipped with two operations (vector addition and scalar multiplication) satisfying these **8 axioms**:

| # | Axiom | Statement |
|---|-------|-----------|
| 1 | Commutativity of addition | **u** + **v** = **v** + **u** |
| 2 | Associativity of addition | (**u** + **v**) + **w** = **u** + (**v** + **w**) |
| 3 | Additive identity | ∃ **0** such that **v** + **0** = **v** |
| 4 | Additive inverse | ∀ **v**, ∃ (−**v**) such that **v** + (−**v**) = **0** |
| 5 | Compatibility of scalar mult. | a(b**v**) = (ab)**v** |
| 6 | Identity element of scalar mult. | 1**v** = **v** |
| 7 | Distributivity over vector add. | a(**u** + **v**) = a**u** + a**v** |
| 8 | Distributivity over scalar add. | (a + b)**v** = a**v** + b**v** |

**Example vector spaces:**
- ℝⁿ with standard addition and scalar multiplication
- The set of all m×n matrices
- The set of all continuous functions on [a, b]

### 2.7 Linear Combination

A vector **v** is a **linear combination** of vectors **v₁**, **v₂**, ..., **vₖ** if:

> **v** = c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ**

for some scalars c₁, c₂, ..., cₖ.

### 2.8 Span

The **span** of a set of vectors {**v₁**, **v₂**, ..., **vₖ**} is the set of **all possible linear combinations**:

> span{**v₁**, ..., **vₖ**} = {c₁**v₁** + ... + cₖ**vₖ** | c₁, ..., cₖ ∈ F}

### 2.9 Linear Independence

Vectors **v₁**, **v₂**, ..., **vₖ** are **linearly independent** if:

> c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0** implies c₁ = c₂ = ... = cₖ = 0

In other words, no vector in the set can be written as a linear combination of the others.

### 2.10 Basis

A **basis** for a vector space V is a set of vectors that is:
1. **Linearly independent**, AND
2. **Spans** V

The number of vectors in any basis of V is the **dimension** of V.

**Standard basis for ℝ³:**

> **e₁** = (1, 0, 0), **e₂** = (0, 1, 0), **e₃** = (0, 0, 1)

Any vector (a, b, c) in ℝ³ can be written as:

> (a, b, c) = a**e₁** + b**e₂** + c**e₃**

---

## 3. Key Formulas with Derivations

### 3.1 Euclidean Norm (L2 Norm)

**Formula:**

> ‖**v**‖ = √(v₁² + v₂² + ... + vₙ²) = √(**v** · **v**)

**Derivation from the Pythagorean theorem:**

In 2D, a vector **v** = (v₁, v₂) forms a right triangle with legs v₁ and v₂. By the Pythagorean theorem:

> length² = v₁² + v₂²
> length = √(v₁² + v₂²)

Extending to 3D: the vector (v₁, v₂, v₃) first projects onto the xy-plane with length √(v₁² + v₂²), then forms another right triangle with v₃:

> length² = (√(v₁² + v₂²))² + v₃² = v₁² + v₂² + v₃²

This pattern generalizes to n dimensions by induction:

> ‖**v**‖ = √(Σᵢ vᵢ²)

**Connection to dot product:**

> **v** · **v** = v₁·v₁ + v₂·v₂ + ... + vₙ·vₙ = v₁² + v₂² + ... + vₙ²

Therefore:

> ‖**v**‖ = √(**v** · **v**)

### 3.2 Dot Product — Geometric Form

**Formula:**

> **u** · **v** = ‖**u**‖ · ‖**v**‖ · cos(θ)

where θ is the angle between **u** and **v**.

**Derivation using the law of cosines:**

Consider vectors **u** and **v** with angle θ between them. The vector **u** − **v** completes a triangle. By the law of cosines:

> Step 1: ‖**u** − **v**‖² = ‖**u**‖² + ‖**v**‖² − 2‖**u**‖‖**v**‖cos(θ)

Expand the left side using the dot product:

> Step 2: ‖**u** − **v**‖² = (**u** − **v**) · (**u** − **v**)
> Step 3: = **u**·**u** − 2(**u**·**v**) + **v**·**v**
> Step 4: = ‖**u**‖² − 2(**u**·**v**) + ‖**v**‖²

Set Step 1 equal to Step 4:

> Step 5: ‖**u**‖² − 2(**u**·**v**) + ‖**v**‖² = ‖**u**‖² + ‖**v**‖² − 2‖**u**‖‖**v**‖cos(θ)

Cancel ‖**u**‖² and ‖**v**‖² from both sides:

> Step 6: −2(**u**·**v**) = −2‖**u**‖‖**v**‖cos(θ)

Divide by −2:

> Step 7: **u**·**v** = ‖**u**‖ · ‖**v**‖ · cos(θ) ∎

**Key implications:**
- If **u** · **v** = 0, then cos(θ) = 0, so θ = 90° → vectors are **orthogonal**
- If **u** · **v** > 0, the angle is acute (< 90°)
- If **u** · **v** < 0, the angle is obtuse (> 90°)

### 3.3 Unit Vector

**Formula:**

> **û** = **v** / ‖**v**‖

**Derivation:**

We want a vector in the same direction as **v** but with magnitude 1. Scaling **v** by scalar *c*:

> ‖c**v**‖ = |c| · ‖**v**‖

Set this equal to 1:

> |c| · ‖**v**‖ = 1
> |c| = 1 / ‖**v**‖

Choosing c > 0 (same direction):

> **û** = (1/‖**v**‖) · **v** = **v** / ‖**v**‖

**Verification:** ‖**û**‖ = ‖**v** / ‖**v**‖‖ = ‖**v**‖ / ‖**v**‖ = 1 ✓

### 3.4 Vector Projection

The **scalar projection** of **u** onto **v**:

> comp_**v**(**u**) = (**u** · **v**) / ‖**v**‖

The **vector projection** of **u** onto **v**:

> proj_**v**(**u**) = [(**u** · **v**) / (**v** · **v**)] · **v**

**Derivation:**

We want the component of **u** in the direction of **v**. The projection must be some scalar multiple of **v**:

> proj_**v**(**u**) = c**v**

The residual (**u** − c**v**) must be orthogonal to **v**:

> Step 1: (**u** − c**v**) · **v** = 0
> Step 2: **u**·**v** − c(**v**·**v**) = 0
> Step 3: c = (**u**·**v**) / (**v**·**v**)

Therefore:

> proj_**v**(**u**) = [(**u**·**v**) / (**v**·**v**)] · **v** ∎

### 3.5 Cosine Similarity

**Formula:**

> cos(θ) = (**u** · **v**) / (‖**u**‖ · ‖**v**‖)

This is directly rearranged from the geometric dot product formula (Section 3.2).

**Range:** −1 ≤ cos(θ) ≤ 1

- cos(θ) = 1 → identical direction
- cos(θ) = 0 → orthogonal (unrelated)
- cos(θ) = −1 → opposite direction

**GPU/ML significance:** Cosine similarity is the standard measure of similarity between embedding vectors in NLP, recommendation systems, and retrieval-augmented generation (RAG).

---

## 4. Worked Examples — Small Scale

### Example 4.1: Vector Addition and Scalar Multiplication

**Problem:** Given **u** = (3, −1, 4) and **v** = (2, 5, −2), compute **w** = 2**u** + 3**v**.

**Solution:**

> Step 1: Compute 2**u**
> 2**u** = 2·(3, −1, 4) = (2·3, 2·(−1), 2·4) = (6, −2, 8)

> Step 2: Compute 3**v**
> 3**v** = 3·(2, 5, −2) = (3·2, 3·5, 3·(−2)) = (6, 15, −6)

> Step 3: Add component-wise
> **w** = (6, −2, 8) + (6, 15, −6) = (6+6, −2+15, 8+(−6)) = **(12, 13, 2)**

### Example 4.2: Dot Product (Algebraic)

**Problem:** Compute **u** · **v** for **u** = (4, −3, 2) and **v** = (1, 5, −6).

**Solution:**

> Step 1: Multiply corresponding components
> u₁v₁ = 4 · 1 = 4
> u₂v₂ = (−3) · 5 = −15
> u₃v₃ = 2 · (−6) = −12

> Step 2: Sum all products
> **u** · **v** = 4 + (−15) + (−12) = 4 − 15 − 12 = **−23**

### Example 4.3: Euclidean Norm

**Problem:** Find ‖**v**‖ for **v** = (3, 4, 12).

**Solution:**

> Step 1: Square each component
> 3² = 9
> 4² = 16
> 12² = 144

> Step 2: Sum the squares
> 9 + 16 + 144 = 169

> Step 3: Take the square root
> ‖**v**‖ = √169 = **13**

### Example 4.4: Unit Vector

**Problem:** Find the unit vector in the direction of **v** = (3, 4, 12).

**Solution:**

> Step 1: From Example 4.3, ‖**v**‖ = 13

> Step 2: Divide each component by the norm
> **û** = (3/13, 4/13, 12/13)

> Step 3: Verify
> ‖**û**‖ = √((3/13)² + (4/13)² + (12/13)²)
> = √(9/169 + 16/169 + 144/169)
> = √(169/169)
> = √1 = 1 ✓

> **û** = **(3/13, 4/13, 12/13) ≈ (0.2308, 0.3077, 0.9231)**

### Example 4.5: Angle Between Vectors

**Problem:** Find the angle between **u** = (1, 0, 1) and **v** = (0, 1, 1).

**Solution:**

> Step 1: Compute **u** · **v**
> **u** · **v** = 1·0 + 0·1 + 1·1 = 0 + 0 + 1 = 1

> Step 2: Compute ‖**u**‖
> ‖**u**‖ = √(1² + 0² + 1²) = √(1 + 0 + 1) = √2

> Step 3: Compute ‖**v**‖
> ‖**v**‖ = √(0² + 1² + 1²) = √(0 + 1 + 1) = √2

> Step 4: Apply cosine similarity formula
> cos(θ) = (**u**·**v**) / (‖**u**‖ · ‖**v**‖) = 1 / (√2 · √2) = 1/2

> Step 5: Find the angle
> θ = arccos(1/2) = **60° = π/3 radians**

### Example 4.6: Vector Projection

**Problem:** Project **u** = (3, 4) onto **v** = (1, 0).

**Solution:**

> Step 1: Compute **u** · **v**
> **u** · **v** = 3·1 + 4·0 = 3

> Step 2: Compute **v** · **v**
> **v** · **v** = 1·1 + 0·0 = 1

> Step 3: Compute the scalar coefficient
> c = (**u**·**v**) / (**v**·**v**) = 3/1 = 3

> Step 4: Compute the projection vector
> proj_**v**(**u**) = 3 · (1, 0) = **(3, 0)**

This makes geometric sense: projecting (3, 4) onto the x-axis gives (3, 0) — the x-component.

### Example 4.7: Linear Independence Check

**Problem:** Are **v₁** = (1, 2, 3), **v₂** = (4, 5, 6), **v₃** = (7, 8, 9) linearly independent?

**Solution:**

> Step 1: Set up c₁**v₁** + c₂**v₂** + c₃**v₃** = **0**
> c₁(1,2,3) + c₂(4,5,6) + c₃(7,8,9) = (0,0,0)

> Step 2: Write as system of equations
> c₁ + 4c₂ + 7c₃ = 0   ... (i)
> 2c₁ + 5c₂ + 8c₃ = 0   ... (ii)
> 3c₁ + 6c₂ + 9c₃ = 0   ... (iii)

> Step 3: (ii) − 2·(i):
> (2c₁ − 2c₁) + (5c₂ − 8c₂) + (8c₃ − 14c₃) = 0
> −3c₂ − 6c₃ = 0
> c₂ = −2c₃   ... (iv)

> Step 4: (iii) − 3·(i):
> (3c₁ − 3c₁) + (6c₂ − 12c₂) + (9c₃ − 21c₃) = 0
> −6c₂ − 12c₃ = 0
> c₂ = −2c₃   ... (v)

> Step 5: (iv) and (v) are the same equation — free variable exists
> Let c₃ = 1, then c₂ = −2
> From (i): c₁ + 4(−2) + 7(1) = 0 → c₁ − 8 + 7 = 0 → c₁ = 1

> Step 6: Verify: 1·(1,2,3) + (−2)·(4,5,6) + 1·(7,8,9)
> = (1,2,3) + (−8,−10,−12) + (7,8,9)
> = (1−8+7, 2−10+8, 3−12+9) = (0, 0, 0) ✓

> **Result:** Non-trivial solution (c₁, c₂, c₃) = (1, −2, 1) exists → **Linearly dependent**
>
> Geometric insight: **v₃** = 2**v₂** − **v₁**, meaning the three vectors are coplanar.

---

## 5. Worked Examples — GPU Scale

### Example 5.1: Dot Product of 4096-Dimensional Vectors

**Context:** In transformer models (GPT, BERT, LLaMA), embedding vectors are commonly 4096-dimensional. Computing attention scores requires the dot product of query and key vectors.

**Problem:** Conceptually compute **q** · **k** where **q**, **k** ∈ ℝ⁴⁰⁹⁶.

**Solution (conceptual + computational):**

> Step 1: Element-wise multiplication → produces 4096 intermediate products
> p[i] = q[i] * k[i],  for i = 0, 1, ..., 4095

> Step 2: Sum all 4096 products (reduction)
> dot = p[0] + p[1] + p[2] + ... + p[4095]

**Parallelism analysis:**

> Step 3: Step 1 is embarrassingly parallel — all 4096 multiplications are independent
> On a GPU: assign one thread per element, all 4096 execute simultaneously

> Step 4: Step 2 is a **reduction** — requires log₂(4096) = 12 sequential steps
> Round 1: 2048 threads each add 2 values → 2048 partial sums
> Round 2: 1024 threads → 1024 partial sums
> ...
> Round 12: 1 thread → final scalar result

> Step 5: Total arithmetic operations = 4096 multiplications + 4095 additions = 8191
> Sequential time: 8191 steps
> Parallel time: 1 step (multiply) + 12 steps (reduce) = 13 steps
> **Speedup: ~630x**

### Example 5.2: Batch Cosine Similarity

**Context:** In retrieval systems, you compare a query embedding against a database of N document embeddings.

**Problem:** Compute cosine similarity between query **q** ∈ ℝ⁷⁶⁸ and N = 100,000 document vectors **d₁**, ..., **d_N** ∈ ℝ⁷⁶⁸.

**Solution:**

> Step 1: Pre-compute ‖**q**‖ (once)
> ‖**q**‖ = √(Σᵢ qᵢ²)   — 768 multiplications + 767 additions + 1 sqrt

> Step 2: Pre-compute ‖**dⱼ**‖ for all j = 1..100,000
> These can be cached at index time

> Step 3: For each document j, compute **q** · **dⱼ**
> Each dot product: 768 multiplications + 767 additions

> Step 4: Divide: cos(θⱼ) = (**q** · **dⱼ**) / (‖**q**‖ · ‖**dⱼ**‖)

**GPU strategy:**

> Step 5: Total dot products needed: 100,000
> Each dot product: 1,535 FLOPs
> Total: ~153.5 million FLOPs
> Modern GPU (A100): ~312 TFLOPS (FP16)
> Theoretical time: 153.5M / 312T ≈ 0.0005 ms
> (Memory bandwidth is the actual bottleneck, not compute)

### Example 5.3: Vector Addition at Scale — Gradient Accumulation

**Context:** During neural network training, gradients are vectors with millions of dimensions. Gradient accumulation adds gradient vectors across micro-batches.

**Problem:** Accumulate 8 gradient vectors, each with 175 billion parameters (GPT-3 scale).

**Solution:**

> Step 1: Each gradient is a vector in ℝ^(175,000,000,000)

> Step 2: Accumulation = element-wise addition:
> **g_total** = **g₁** + **g₂** + **g₃** + ... + **g₈**

> Step 3: Each component is independent:
> g_total[i] = g₁[i] + g₂[i] + ... + g₈[i]   for each i

> Step 4: Memory requirements:
> 175B parameters × 4 bytes (FP32) = 700 GB per gradient vector
> 8 gradients = 5.6 TB → must be distributed across multiple GPUs

> Step 5: On 8× A100 (80GB each) = 640 GB total
> Each GPU holds ~21.875B parameters worth of gradient
> Addition is local — no cross-GPU communication needed for this step

---

## 6. CUDA Kernel Design Connections

### 6.1 Vector Addition Kernel

Vector addition maps **one thread per element** — the simplest possible CUDA pattern.

```cpp
// CUDA kernel: vector addition
// Each thread computes one element of the output
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check (N may not be a multiple of block size)
    if (i < N) {
        C[i] = A[i] + B[i];  // One addition per thread
    }
}

// Host launch configuration
// blockSize: threads per block (typically 256 or 512)
// gridSize: number of blocks needed to cover all N elements
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
```

**Why this works:**
- No thread depends on another thread's result
- Memory access is **coalesced** — consecutive threads read consecutive addresses
- Pattern: `thread i reads A[i] and B[i], writes C[i]`

### 6.2 Scalar Multiplication Kernel

Identical pattern to vector addition — one thread per element.

```cpp
__global__ void scalarMul(const float* V, float scalar, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = scalar * V[i];
    }
}
```

### 6.3 Dot Product Kernel (Reduction Pattern)

The dot product is fundamentally different — it requires **inter-thread communication**.

```cpp
__global__ void dotProduct(const float* A, const float* B, float* result, int N) {
    // Shared memory for block-level reduction
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: Each thread computes one multiplication
    sdata[tid] = (i < N) ? A[i] * B[i] : 0.0f;
    __syncthreads();  // Barrier: wait for all threads in block
    
    // Step 2: Tree reduction within the block
    // Each iteration halves the number of active threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // Barrier after each reduction step
    }
    
    // Step 3: Thread 0 of each block writes its block's partial sum
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}
```

**Key concepts illustrated:**
- **Shared memory (`__shared__`):** Fast on-chip memory visible to all threads in a block
- **`__syncthreads()`:** Barrier synchronization — all threads must reach this point before any proceed
- **Tree reduction:** log₂(blockSize) steps to reduce blockSize values to 1
- **`atomicAdd`:** Thread-safe addition for combining results across blocks

### 6.4 Norm Kernel

Computing ‖**v**‖ is a dot product of **v** with itself, plus a square root:

```cpp
// Conceptual: reuse dot product kernel
// Step 1: dot = dotProduct(V, V, N)    → gives ‖v‖²
// Step 2: norm = sqrtf(dot)            → gives ‖v‖
```

### 6.5 Memory Access Patterns Summary

| Operation | Pattern | Parallelism | Communication |
|-----------|---------|-------------|---------------|
| Vector Addition | 1 thread/element | Embarrassingly parallel | None |
| Scalar Multiply | 1 thread/element | Embarrassingly parallel | None |
| Dot Product | Multiply + reduce | Parallel multiply, tree reduce | Within block + across blocks |
| Norm | Dot product + sqrt | Same as dot product | Same as dot product |
| Cosine Similarity | 2 norms + dot product | Composed | Same as dot product |

### 6.6 Thinking About Memory

For GPU programming, understanding data layout is as important as understanding math:

```
Vector of N floats in memory:
Address:  [0x00] [0x04] [0x08] [0x0C] [0x10] ...
Value:     v[0]   v[1]   v[2]   v[3]   v[4]  ...
Thread:    T0     T1     T2     T3     T4    ...

Coalesced access: Thread i reads address base + i*sizeof(float)
→ GPU memory controller combines into a single wide transaction
→ Maximum memory bandwidth utilization
```

---

## 7. Practice Problems with Full Solutions

### Problem 7.1: Vector Arithmetic

**Problem:** Given **a** = (2, −1, 3, 0, 5) and **b** = (−1, 4, 2, −3, 1), compute:
1. **a** + **b**
2. **a** − **b**
3. 3**a** − 2**b**
4. ‖**a**‖
5. **a** · **b**

---

**Solution:**

**Part 1: a + b**

> Step 1: Add component-wise
> (2+(−1), (−1)+4, 3+2, 0+(−3), 5+1)

> Step 2: Simplify
> **a** + **b** = **(1, 3, 5, −3, 6)**

**Part 2: a − b**

> Step 1: Subtract component-wise
> (2−(−1), (−1)−4, 3−2, 0−(−3), 5−1)

> Step 2: Simplify
> **a** − **b** = **(3, −5, 1, 3, 4)**

**Part 3: 3a − 2b**

> Step 1: Compute 3**a**
> 3(2, −1, 3, 0, 5) = (6, −3, 9, 0, 15)

> Step 2: Compute 2**b**
> 2(−1, 4, 2, −3, 1) = (−2, 8, 4, −6, 2)

> Step 3: Subtract
> (6−(−2), (−3)−8, 9−4, 0−(−6), 15−2)

> Step 4: Simplify
> 3**a** − 2**b** = **(8, −11, 5, 6, 13)**

**Part 4: ‖a‖**

> Step 1: Square each component
> 2² = 4,  (−1)² = 1,  3² = 9,  0² = 0,  5² = 25

> Step 2: Sum
> 4 + 1 + 9 + 0 + 25 = 39

> Step 3: Square root
> ‖**a**‖ = **√39 ≈ 6.245**

**Part 5: a · b**

> Step 1: Multiply corresponding components
> 2·(−1) = −2
> (−1)·4 = −4
> 3·2 = 6
> 0·(−3) = 0
> 5·1 = 5

> Step 2: Sum
> −2 + (−4) + 6 + 0 + 5 = **5**

---

### Problem 7.2: Orthogonality and Angle

**Problem:** Determine if **u** = (1, −2, 3) and **v** = (4, 5, 2) are orthogonal. If not, find the angle between them.

---

**Solution:**

> Step 1: Compute **u** · **v**
> 1·4 + (−2)·5 + 3·2 = 4 − 10 + 6 = 0

> Step 2: Since **u** · **v** = 0, the vectors are **orthogonal** ✓

> The angle between them is **θ = 90° = π/2 radians**

---

### Problem 7.3: Projection

**Problem:** Find the projection of **u** = (4, 3, 1) onto **v** = (2, −1, 2).

---

**Solution:**

> Step 1: Compute **u** · **v**
> 4·2 + 3·(−1) + 1·2 = 8 − 3 + 2 = 7

> Step 2: Compute **v** · **v**
> 2·2 + (−1)·(−1) + 2·2 = 4 + 1 + 4 = 9

> Step 3: Compute scalar coefficient
> c = 7/9

> Step 4: Compute projection vector
> proj_**v**(**u**) = (7/9)(2, −1, 2) = **(14/9, −7/9, 14/9) ≈ (1.556, −0.778, 1.556)**

> Step 5: Verify orthogonality of residual
> Residual: **u** − proj = (4 − 14/9, 3 + 7/9, 1 − 14/9) = (22/9, 34/9, −5/9)
> Check: (22/9)·2 + (34/9)·(−1) + (−5/9)·2 = 44/9 − 34/9 − 10/9 = 0/9 = 0 ✓

---

### Problem 7.4: Linear Independence

**Problem:** Are the vectors **v₁** = (1, 0, 2), **v₂** = (0, 1, −1), **v₃** = (2, 1, 3) linearly independent?

---

**Solution:**

> Step 1: Set up c₁**v₁** + c₂**v₂** + c₃**v₃** = **0**
> c₁(1,0,2) + c₂(0,1,−1) + c₃(2,1,3) = (0,0,0)

> Step 2: System of equations
> c₁ + 0c₂ + 2c₃ = 0   ... (i)
> 0c₁ + c₂ + c₃ = 0     ... (ii)
> 2c₁ − c₂ + 3c₃ = 0    ... (iii)

> Step 3: From (i): c₁ = −2c₃

> Step 4: From (ii): c₂ = −c₃

> Step 5: Check (iii): 2(−2c₃) − (−c₃) + 3c₃ = −4c₃ + c₃ + 3c₃ = 0c₃ = 0

> Step 6: Equation (iii) is satisfied for ALL values of c₃
> Set c₃ = 1 → c₁ = −2, c₂ = −1

> Step 7: Verify: −2(1,0,2) + (−1)(0,1,−1) + 1(2,1,3)
> = (−2,0,−4) + (0,−1,1) + (2,1,3)
> = (−2+0+2, 0−1+1, −4+1+3) = (0, 0, 0) ✓

> **Result: Linearly dependent.** The relation is **v₃** = 2**v₁** + **v₂**.

---

### Problem 7.5: Basis Verification

**Problem:** Show that **e₁** = (1, 1, 0), **e₂** = (1, 0, 1), **e₃** = (0, 1, 1) form a basis for ℝ³.

---

**Solution:**

We must show: (a) they span ℝ³, and (b) they are linearly independent.

**Part (a): Linear Independence**

> Step 1: Set up c₁**e₁** + c₂**e₂** + c₃**e₃** = **0**
> c₁(1,1,0) + c₂(1,0,1) + c₃(0,1,1) = (0,0,0)

> Step 2: System of equations
> c₁ + c₂ = 0          ... (i)
> c₁ + c₃ = 0          ... (ii)
> c₂ + c₃ = 0          ... (iii)

> Step 3: From (i): c₂ = −c₁

> Step 4: From (ii): c₃ = −c₁

> Step 5: Substitute into (iii): (−c₁) + (−c₁) = 0 → −2c₁ = 0 → c₁ = 0

> Step 6: Therefore c₁ = c₂ = c₃ = 0 → **linearly independent** ✓

**Part (b): Span ℝ³**

> Step 7: Since we have 3 linearly independent vectors in ℝ³, and dim(ℝ³) = 3,
> they automatically form a basis and span all of ℝ³.

> Step 8: Explicit verification — express (a, b, c) in terms of the basis:
> c₁(1,1,0) + c₂(1,0,1) + c₃(0,1,1) = (a, b, c)
>
> c₁ + c₂ = a
> c₁ + c₃ = b
> c₂ + c₃ = c
>
> Adding all three: 2(c₁ + c₂ + c₃) = a + b + c → c₁ + c₂ + c₃ = (a+b+c)/2
>
> c₃ = (a+b+c)/2 − a = (−a+b+c)/2
> c₂ = (a+b+c)/2 − b = (a−b+c)/2
> c₁ = (a+b+c)/2 − c = (a+b−c)/2
>
> A unique solution exists for every (a, b, c) → **spans ℝ³** ✓

> **Result: {e₁, e₂, e₃} is a basis for ℝ³.** ∎

---

### Problem 7.6: CUDA Thread Mapping

**Problem:** You need to add two vectors of length N = 10,000. Your CUDA block size is 256. How many blocks do you need? What does the last block look like?

---

**Solution:**

> Step 1: Calculate number of blocks
> gridSize = ⌈N / blockSize⌉ = ⌈10000 / 256⌉

> Step 2: Compute the division
> 10000 / 256 = 39.0625

> Step 3: Ceiling
> gridSize = **40 blocks**

> Step 4: Total threads launched
> 40 × 256 = 10,240 threads

> Step 5: Threads in last block
> All 256 threads launch, but only 10000 − 39×256 = 10000 − 9984 = **16 threads do useful work**

> Step 6: The remaining 256 − 16 = 240 threads in the last block hit the bounds check
> `if (i < N)` and exit without performing computation

> **Answer: 40 blocks. The last block has 256 threads launched, but only 16 active.**

---

### Problem 7.7: Cosine Similarity in Practice

**Problem:** Two word embedding vectors (simplified to 5D):
- "king" → **k** = (0.8, 0.2, −0.5, 0.9, 0.1)
- "queen" → **q** = (0.7, 0.3, −0.4, 0.85, 0.15)

Compute their cosine similarity.

---

**Solution:**

> Step 1: Compute **k** · **q**
> 0.8·0.7 = 0.56
> 0.2·0.3 = 0.06
> (−0.5)·(−0.4) = 0.20
> 0.9·0.85 = 0.765
> 0.1·0.15 = 0.015
> **k** · **q** = 0.56 + 0.06 + 0.20 + 0.765 + 0.015 = 1.600

> Step 2: Compute ‖**k**‖
> 0.8² + 0.2² + 0.5² + 0.9² + 0.1² = 0.64 + 0.04 + 0.25 + 0.81 + 0.01 = 1.75
> ‖**k**‖ = √1.75 ≈ 1.3229

> Step 3: Compute ‖**q**‖
> 0.7² + 0.3² + 0.4² + 0.85² + 0.15² = 0.49 + 0.09 + 0.16 + 0.7225 + 0.0225 = 1.485
> ‖**q**‖ = √1.485 ≈ 1.2187

> Step 4: Cosine similarity
> cos(θ) = 1.600 / (1.3229 × 1.2187) = 1.600 / 1.6124 ≈ **0.9923**

> **Interpretation:** cos(θ) ≈ 0.99 means "king" and "queen" are **very similar** in this
> embedding space — nearly parallel vectors. This reflects that they share most semantic
> properties (royalty, authority, high status) and differ primarily in one dimension (gender).

---

### Problem 7.8: Reduction Tree Analysis

**Problem:** You have a block of 512 threads performing a dot product reduction. How many reduction steps are needed? After step 3, how many active threads remain?

---

**Solution:**

> Step 1: Number of reduction steps = log₂(512) = **9 steps**

> Step 2: Active threads per step:
> Step 0 (multiply): 512 threads active
> Step 1 (first reduce): 256 threads active (each adds 2 values)
> Step 2: 128 threads active
> Step 3: 64 threads active
> Step 4: 32 threads active (one warp — can use warp-level primitives)
> Step 5: 16 threads
> Step 6: 8 threads
> Step 7: 4 threads
> Step 8: 2 threads
> Step 9: 1 thread → final result

> **After step 3: 64 threads remain active.**

> **Observation:** Thread utilization drops rapidly. At step 5, only 16/512 = 3.125% of
> threads are working. This is why optimized reduction kernels use techniques like
> **warp shuffle instructions** and **multi-element-per-thread loading** to improve efficiency.

---

*Assad's 52-Week NVIDIA Curriculum · Week 1: Linear Algebra I · Phase 1: Math & Systems*
*Saved: mathematics | File: week-01-linear-algebra-i-mathematics.md*
