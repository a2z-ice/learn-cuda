# Week 1: Linear Algebra I — Mathematics
> Phase 1: Math & Systems | Assad's NVIDIA Curriculum | Week 1 of 52
> Topic: Vectors, scalars, dot product, vector spaces, basis vectors

---

## 1. Intuition Before Symbols

### What Is a Vector?

Forget math for a moment. Imagine you are giving someone directions in a city:

> *"Walk 3 blocks east and 4 blocks north."*

That instruction has two pieces: a **direction** and a **magnitude** (amount). That is a vector. You cannot describe it with a single number — you need both components together. A scalar, by contrast, is just a single number: temperature, price, distance. It has magnitude but no direction.

Now think about a weather map. Every point on the map has a wind arrow — pointing somewhere, with some strength. That entire map is a **vector field**: a vector at every point in space. GPU parallelism is, in a deep sense, about processing all of those arrows simultaneously.

### What Is a Dot Product?

Hold both arms out straight in front of you, pointing forward. Now rotate one arm 90° to the side. Ask: *"How much of that arm's direction is in common with the other?"* The answer is zero — they share nothing. Now point them both forward again: they share everything.

The dot product measures exactly this: **how much two vectors point in the same direction**. It collapses two arrows into a single number that encodes their agreement.

This is not an abstract concept. In a neural network, every neuron computes a dot product: it asks *"how much does my input agree with what I'm looking for?"* A high dot product means strong activation. Zero means this neuron ignores that input entirely. The entire intelligence of a language model is encoded in billions of these agreements.

### What Is a Vector Space?

Imagine you have a set of LEGO bricks. The rules of a vector space are the rules of how those bricks combine: you can always stick two bricks together (addition stays in the set), you can scale a brick (multiplication stays in the set), and there's an empty brick that changes nothing (zero vector). The 8 axioms are just formalizing that these rules are consistent and well-behaved.

### What Are Basis Vectors?

Every color on your screen is made from three values: Red, Green, Blue. Any color in existence can be described as some mixture of these three. Red, Green, Blue are the **basis** of color space — the minimum set of independent building blocks that can construct everything else. Basis vectors are exactly this for mathematical space: the irreducible directions from which every other direction can be built.

---

## 2. Formal Definition — Build It Step by Step

### Scalars

A **scalar** is an element of a field F — for our purposes, a real number in ℝ. Examples: `λ = 3`, `α = -0.5`, `c = 2.718`.

It has magnitude only. No direction.

### Vectors

A **vector** `v` in ℝⁿ is an ordered tuple of n real numbers:

```
v = [v₁, v₂, v₃, ..., vₙ]
```

where each `vᵢ ∈ ℝ` is called a **component** (or coordinate) of the vector. The number n is the **dimension** of the vector.

- `v ∈ ℝ²`: a point in a 2D plane — `v = [3, -2]`
- `v ∈ ℝ³`: a point in 3D space — `v = [1, 4, -7]`
- `v ∈ ℝ⁴⁰⁹⁶`: a token embedding in LLaMA-7B

Vectors are written as column vectors in formal math, or row vectors in code (arrays). We use row notation throughout for readability.

### Vector Operations — Built From the Analogy

**Addition:** Combine two arrows tip-to-tail.
```
u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
```
Component-wise. If `u = [1, 2]` and `v = [3, -1]`, then `u + v = [4, 1]`.

**Scalar multiplication:** Stretch or shrink an arrow.
```
λ · v = [λv₁, λv₂, ..., λvₙ]
```
If `λ = 3` and `v = [1, 2]`, then `3v = [3, 6]`. Direction preserved, magnitude scaled.

**Subtraction:** Point from the tip of v to the tip of u.
```
u - v = [u₁-v₁, u₂-v₂, ..., uₙ-vₙ]
```

### Vector Spaces — The 8 Axioms

A **vector space** V over ℝ is a set of objects (vectors) with two operations — addition (+) and scalar multiplication (·) — satisfying these 8 axioms for all `u, v, w ∈ V` and all `λ, μ ∈ ℝ`:

| # | Axiom | Name |
|---|---|---|
| 1 | `u + v ∈ V` | Closure under addition |
| 2 | `u + v = v + u` | Commutativity |
| 3 | `(u+v)+w = u+(v+w)` | Associativity |
| 4 | `∃ 0 ∈ V : v + 0 = v` | Zero vector exists |
| 5 | `∃ -v : v + (-v) = 0` | Additive inverse exists |
| 6 | `λv ∈ V` | Closure under scalar multiplication |
| 7 | `λ(u+v) = λu + λv` | Distributive (vector addition) |
| 8 | `(λ+μ)v = λv + μv` | Distributive (scalar addition) |

ℝⁿ satisfies all 8 axioms for all n ≥ 1. This is the space we work in for all of ML.

### Linear Independence, Span, and Basis

**Span:** The set of all possible linear combinations of a set of vectors S = {v₁, v₂, ..., vₖ}:
```
span(S) = { c₁v₁ + c₂v₂ + ... + cₖvₖ  |  c₁,...,cₖ ∈ ℝ }
```
The span of S is every vector you can reach by mixing the vectors in S.

**Linear independence:** Vectors {v₁, ..., vₖ} are linearly independent if the only solution to:
```
c₁v₁ + c₂v₂ + ... + cₖvₖ = 0
```
is `c₁ = c₂ = ... = cₖ = 0`. No vector in the set can be built from the others.

**Basis:** A set B = {b₁, ..., bₙ} is a basis for V if it is:
1. Linearly independent
2. Spans V — every vector in V is a linear combination of B

**Standard basis for ℝ³:**
```
e₁ = [1, 0, 0]
e₂ = [0, 1, 0]
e₃ = [0, 0, 1]
```
Any `v = [a, b, c] ∈ ℝ³` satisfies `v = a·e₁ + b·e₂ + c·e₃`. The components of a vector are exactly the coefficients in the standard basis expansion.

---

## 3. Key Formulas — Derive Every One

### Formula 1: The Dot Product

**Statement:**
```
u · v = Σᵢ₌₁ⁿ uᵢvᵢ  =  u₁v₁ + u₂v₂ + ... + uₙvₙ
```
where `u, v ∈ ℝⁿ`. The result is a **scalar** — not a vector.

**Derivation from geometric definition:**

We define the dot product geometrically as:
```
u · v = ||u|| · ||v|| · cos(θ)
```
where `||u||` is the magnitude of u and θ is the angle between u and v.

Apply the **Law of Cosines** to the triangle formed by u, v, and u−v:
```
||u - v||² = ||u||² + ||v||² - 2||u||·||v||·cos(θ)
```

Expand `||u - v||²` component-wise:
```
||u - v||² = Σᵢ (uᵢ - vᵢ)²
           = Σᵢ (uᵢ² - 2uᵢvᵢ + vᵢ²)
           = Σᵢ uᵢ² - 2Σᵢ uᵢvᵢ + Σᵢ vᵢ²
           = ||u||² - 2Σᵢ uᵢvᵢ + ||v||²
```

Substitute into Law of Cosines:
```
||u||² - 2Σᵢ uᵢvᵢ + ||v||² = ||u||² + ||v||² - 2||u||·||v||·cos(θ)
```

Cancel `||u||²` and `||v||²`:
```
-2Σᵢ uᵢvᵢ = -2||u||·||v||·cos(θ)
```

Divide by −2:
```
Σᵢ uᵢvᵢ = ||u||·||v||·cos(θ)
```

Therefore:
```
u · v  =  Σᵢ uᵢvᵢ  =  ||u||·||v||·cos(θ)
```
Both definitions are equivalent. ∎

**What each term represents:**

| Term | Meaning |
|---|---|
| `uᵢvᵢ` | Signed product of the i-th component alignment |
| `Σᵢ uᵢvᵢ` | Total alignment summed across all dimensions |
| `||u||·||v||` | Product of magnitudes — scales the result |
| `cos(θ)` | Pure directional agreement: 1 (same), 0 (perpendicular), −1 (opposite) |

**Special cases:**
```
θ = 0°   → cos(θ) = 1   → u·v = ||u||·||v||   (maximum — fully aligned)
θ = 90°  → cos(θ) = 0   → u·v = 0              (orthogonal — no agreement)
θ = 180° → cos(θ) = -1  → u·v = -||u||·||v||  (opposite — fully anti-aligned)
```

**Computational cost:** n multiplications + n−1 additions = O(n) operations.

---

### Formula 2: Vector Magnitude (L2 Norm)

**Statement:**
```
||v|| = √(v · v) = √(Σᵢ vᵢ²)
```

**Derivation:**

From the dot product with u = v:
```
v · v = Σᵢ vᵢ·vᵢ = Σᵢ vᵢ²
```

Geometrically: `v · v = ||v||·||v||·cos(0°) = ||v||²`, so:
```
||v||² = Σᵢ vᵢ²
||v||  = √(Σᵢ vᵢ²)
```

This is the **Euclidean norm** (L2 norm). It generalizes the Pythagorean theorem to n dimensions.

---

### Formula 3: Unit Vector (Normalization)

**Statement:**
```
v̂ = v / ||v||
```

**Derivation:**

We want a vector in the same direction as v with magnitude 1. Scale by the reciprocal of its magnitude:
```
||v̂|| = ||v / ||v|||| = ||v|| / ||v|| = 1  ✓
```

Direction preserved; only magnitude changes.

**Why it matters for ML:**
Normalizing before dot product isolates directional similarity — this is **cosine similarity**:
```
cos(θ) = (u · v) / (||u||·||v||)  =  û · v̂
```
Layer normalization in transformers is this formula applied to activation vectors, ensuring stable magnitudes across layers regardless of input scale.

---

### Formula 4: Linear Combination

**Statement:**
```
w = c₁v₁ + c₂v₂ + ... + cₖvₖ  =  Σⱼ₌₁ᵏ cⱼvⱼ
```
where `cⱼ ∈ ℝ` are scalars and `vⱼ ∈ ℝⁿ` are vectors.

**Why it matters:** Every neural network neuron computes a linear combination of its inputs:
```
output = Σⱼ wⱼ · xⱼ  =  w · x   (dot product)
```
An entire layer computes m such dot products simultaneously — one per output neuron — which is exactly **matrix-vector multiplication**. Linear combinations are the fundamental operation of all of deep learning.

---

## 4. Worked Example — Small Numbers

### Setup

Let `u = [2, -1, 3]` and `v = [4, 0, -2]` in ℝ³.

### Step 1: Dot Product

```
u · v = u₁v₁ + u₂v₂ + u₃v₃
      = (2)(4) + (-1)(0) + (3)(-2)
      = 8 + 0 + (-6)
      = 2
```

### Step 2: Magnitude of u

```
||u|| = √(u₁² + u₂² + u₃²)
      = √(2² + (-1)² + 3²)
      = √(4 + 1 + 9)
      = √14
      ≈ 3.7417
```

### Step 3: Magnitude of v

```
||v|| = √(4² + 0² + (-2)²)
      = √(16 + 0 + 4)
      = √20
      = 2√5
      ≈ 4.4721
```

### Step 4: Angle Between u and v

```
cos(θ) = (u · v) / (||u|| · ||v||)
        = 2 / (√14 · √20)
        = 2 / √280
        = 2 / (2√70)
        = 1 / √70
        ≈ 1 / 8.3666
        ≈ 0.1195

θ = arccos(0.1195) ≈ 83.1°
```

These vectors point nearly perpendicular to each other — very little directional agreement, reflected in the small dot product (2) relative to their magnitudes.

### Step 5: Unit Vectors

```
û = u / ||u|| = [2, -1, 3] / √14
              = [2/√14, -1/√14, 3/√14]
              = [0.5345, -0.2673, 0.8018]

v̂ = v / ||v|| = [4, 0, -2] / (2√5)
              = [2/√5, 0, -1/√5]
              = [0.8944, 0.0000, -0.4472]
```

Verify magnitude: `||û||² = 0.5345² + 0.2673² + 0.8018² = 0.2857 + 0.0714 + 0.6429 = 1.0` ✓

### Step 6: Basis Decomposition

Express `u = [2, -1, 3]` in the standard basis:
```
u = 2·e₁ + (-1)·e₂ + 3·e₃
  = 2·[1,0,0] + (-1)·[0,1,0] + 3·[0,0,1]
  = [2,0,0] + [0,-1,0] + [0,0,3]
  = [2, -1, 3]  ✓
```

The components of a vector **are** its coefficients in the standard basis expansion. This is not a coincidence — it is the definition of what components mean.

---

## 5. Worked Example — GPU Scale

### From ℝ³ to LLaMA-7B

Everything computed above is identical in structure at GPU scale. Only the dimension n changes.

**LLaMA-7B token embedding:** `d_model = 4096`

Two token embedding vectors:
```
u ∈ ℝ⁴⁰⁹⁶   (embedding for token "transformer")
v ∈ ℝ⁴⁰⁹⁶   (embedding for token "model")

u · v = Σᵢ₌₁⁴⁰⁹⁶ uᵢvᵢ
```
Operations: 4,096 multiplications + 4,095 additions = **8,191 arithmetic operations** per dot product.

### Attention Score Matrix at Full Scale

For one attention head with `d_k = 128`, `seq_len = 2048`:

```
Score matrix: [2048 × 2048] entries
Each entry: one dot product of length 128

Total MACs = seq × seq × d_k
           = 2048 × 2048 × 128
           = 536,870,912
           ≈ 537 million multiply-accumulates
```

Per head. LLaMA-7B: 32 heads × 32 layers = **549 billion MACs in attention alone**.

### What Changes at Scale

| Property | Small Example (ℝ³) | LLaMA-7B (ℝ⁴⁰⁹⁶) |
|---|---|---|
| Vector dimension | 3 | 4,096 |
| Dot product ops | 5 | ~8,192 |
| Memory per vector | 12 bytes | 16,384 bytes (16 KB) |
| Attention matrix (seq=2048) | N/A | 2048×2048 = 16M entries |
| Attention FLOPs (one head, d_k=128) | N/A | ~537 million |
| Time on CPU (single core, 10 GFLOPS) | microseconds | ~54 seconds |
| Time on A100 (312 TFLOPS) | — | ~1.7 ms |

### Memory Bandwidth Implication

At d=4096, a single vector is 16 KB. Loading two vectors for one dot product: 32 KB. An A100 has 2 TB/s memory bandwidth — but must sustain this across billions of dot products per second. This is why **CUDA memory access patterns are not optional** — they determine whether you are compute-bound (good) or bandwidth-bound (bad).

A coalesced warp (32 threads, 32 consecutive floats) moves 128 bytes in one transaction.
A strided warp (32 threads, addresses 512 bytes apart) issues 32 separate transactions — **32× the memory traffic for the same compute**.

---

## 6. Connection to CUDA Kernel Design

### The Direct Mapping

The dot product loop:
```
result = 0
for i in range(n):
    result += u[i] * v[i]
```

Has n multiply-accumulate operations that are **completely independent**. Thread i needs only `u[i]` and `v[i]` — nothing from any other thread. This is the ideal GPU problem.

### Kernel Design: Parallel Reduction

```cpp
// Phase 1: parallel element-wise multiply
__global__ void dot_phase1(const float* u, const float* v, float* partial, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        partial[i] = u[i] * v[i];   // each thread: one multiply
    }
}

// Phase 2: parallel reduction with shared memory
__global__ void dot_reduce(float* partial, float* result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? partial[i] : 0.0f;
    __syncthreads();

    // Tree reduction: halve active threads each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) atomicAdd(result, sdata[0]);
}
```

### Memory Access Pattern: Coalesced vs Strided

This is the most important CUDA concept that flows from vector mathematics.

When 32 threads in a warp access memory:

**Coalesced (ideal):**
```cpp
// Thread 0 → addr 0, Thread 1 → addr 1, ..., Thread 31 → addr 31
float val = u[threadIdx.x + blockIdx.x * blockDim.x];
// Hardware: 1 memory transaction for all 32 threads
```

**Strided (catastrophic):**
```cpp
// Thread 0 → addr 0, Thread 1 → addr 128, ..., Thread 31 → addr 3968
float val = u[(threadIdx.x + blockIdx.x * blockDim.x) * 128];
// Hardware: 32 separate memory transactions — 32× slower
```

The dot product formula `Σᵢ uᵢvᵢ` with consecutive indices is naturally coalesced. This is not accidental — the sequential structure of a vector maps perfectly to the sequential structure of a warp.

### Tiling Strategy for Matrix Multiply (GEMM)

A matrix multiply `C = A × B` is a collection of dot products: `C[i][j] = A[i] · B[:,j]`.

```
Tiling approach:
  Divide output C into tiles (e.g., 32×32)
  For each output tile:
    1. Load tile of A into shared memory  ← one global read, reused by 32 threads
    2. Load tile of B into shared memory  ← one global read, reused by 32 threads
    3. All threads compute partial dot products from shared memory (fast)
    4. Advance tile along K dimension, accumulate
    5. Write final tile to C in global memory

Memory traffic reduction:
  Without tiling: each of 32×32=1024 threads reads 32 floats from global memory
                  = 1024 × 32 = 32,768 global memory reads per tile
  With tiling:    32 floats loaded ONCE into shared, reused 32 times
                  = 32 + 32 = 64 global memory reads per tile
  Reduction: ~512× fewer global memory transactions
```

**The mathematical structure dictates the kernel structure.** Once you understand that matrix multiply is a collection of independent dot products, tiling is the only sensible response to the memory hierarchy. This is the intuition behind `TILE_SIZE` always being a power of 2 — it aligns to warp size (32) and cache line size (128 bytes = 32 floats).

---

## 7. Practice Problems with Full Solutions

---

### Problem 1 — Dot Product and Geometric Interpretation (Warm-Up)

**Problem:**

Let `a = [1, 2, 2]` and `b = [-2, 1, k]` where k is an unknown scalar.

(a) Find the value of k such that a and b are orthogonal.
(b) With that value of k, verify the result using the geometric formula.
(c) What does orthogonality mean in the context of a neural network weight vector?

---

**Solution:**

**(a) Find k for orthogonality:**

Orthogonal means `a · b = 0`:
```
a · b = (1)(-2) + (2)(1) + (2)(k)
      = -2 + 2 + 2k
      = 2k

Set 2k = 0  →  k = 0
```

So `b = [-2, 1, 0]` is orthogonal to `a = [1, 2, 2]`.

**(b) Verify geometrically:**

```
||a|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3

||b|| = √((-2)² + 1² + 0²) = √(4 + 1 + 0) = √5

cos(θ) = (a · b) / (||a|| · ||b||)
        = 0 / (3 · √5)
        = 0

θ = arccos(0) = 90°  ✓
```

**(c) Neural network interpretation:**

If a weight vector `w` and input vector `x` are orthogonal, `w · x = 0` → the neuron produces zero pre-activation. This neuron is **completely unresponsive** to this input pattern — the input carries zero signal in the directions the weight vector is sensitive to. In transformer attention, orthogonal Q and K vectors produce zero attention score → the query token ignores that key token entirely. Orthogonality is the geometric definition of "no relationship."

---

### Problem 2 — Basis Decomposition and Linear Independence (Intermediate)

**Problem:**

Consider vectors `v₁ = [1, 1, 0]`, `v₂ = [0, 1, 1]`, `v₃ = [1, 0, -1]` in ℝ³.

(a) Determine whether {v₁, v₂, v₃} is linearly independent.
(b) If independent, express `w = [3, 4, 1]` as a linear combination of {v₁, v₂, v₃}.
(c) Explain why linear independence matters for the rank of a weight matrix and its effect on GPU compute.

---

**Solution:**

**(a) Test for linear independence:**

Set up `c₁v₁ + c₂v₂ + c₃v₃ = 0`:
```
c₁[1,1,0] + c₂[0,1,1] + c₃[1,0,-1] = [0,0,0]
```

Component equations:
```
c₁ + 0·c₂ + c₃  = 0   →  c₁ + c₃ = 0       (i)
c₁ +   c₂ + 0   = 0   →  c₁ + c₂ = 0       (ii)
0  +   c₂ - c₃  = 0   →  c₂ - c₃ = 0       (iii)
```

From (iii): `c₂ = c₃`

Substitute into (ii): `c₁ + c₃ = 0` → `c₁ = -c₃`

Substitute into (i): `-c₃ + c₃ = 0` → `0 = 0` ✓ for any c₃

This means there are **non-trivial solutions**: let `c₃ = 1`, then `c₁ = -1`, `c₂ = 1`:

**Verify:**
```
(-1)[1,1,0] + (1)[0,1,1] + (1)[1,0,-1]
= [-1,-1,0] + [0,1,1] + [1,0,-1]
= [-1+0+1, -1+1+0, 0+1-1]
= [0, 0, 0]  ✓
```

**Conclusion:** {v₁, v₂, v₃} is **linearly dependent**. Specifically: `v₃ = v₁ - v₂`.

**(b) Express w = [3, 4, 1] as a linear combination:**

Since the set is dependent, we cannot uniquely decompose w — we use any two independent vectors. Use {v₁, v₂} as the basis (they are independent — verify: `(-c₁)[1,1,0] = c₂[0,1,1]` has no solution):

```
c₁v₁ + c₂v₂ = [3, 4, 1]

c₁(1) + c₂(0) = 3   →  c₁ = 3
c₁(1) + c₂(1) = 4   →  3 + c₂ = 4  →  c₂ = 1
c₁(0) + c₂(1) = 1   →  c₂ = 1  ✓

w = 3·v₁ + 1·v₂  = 3[1,1,0] + [0,1,1] = [3,3,0] + [0,1,1] = [3,4,1]  ✓
```

**(c) Connection to matrix rank and GPU compute:**

A weight matrix W ∈ ℝ^{m×n} has **rank r** = the number of linearly independent rows (or columns). If `r < min(m,n)`, the matrix is **rank-deficient**: it maps inputs into a lower-dimensional subspace and some of its rows/columns are redundant.

**GPU compute implication:** Suppose W is a 4096×4096 matrix but has rank 512. Then:
- Stored as full matrix: 4096 × 4096 × 4 bytes = **67 MB** VRAM
- Full matmul: 4096 × 4096 × 4096 = **68 billion MACs**
- But only 512 independent directions contribute — 87.5% of the compute is redundant

**LoRA (Low-Rank Adaptation)** exploits this: instead of fine-tuning W directly, it learns `W + AB` where `A ∈ ℝ^{4096×r}` and `B ∈ ℝ^{r×4096}` with `r = 16`. Fine-tuning cost: `2 × 4096 × 16 × 4 = 524 KB` VRAM and proportionally fewer MACs. NVIDIA TensorRT uses similar low-rank compression for model optimization at inference time.

---

### Problem 3 — Dot Product at Scale, NVIDIA Interview Level

**Problem:**

You are implementing attention score computation for a single transformer head:

```
scores = Q · Kᵀ / √d_k
```

where `Q ∈ ℝ^{seq×d_k}`, `K ∈ ℝ^{seq×d_k}`, `d_k = 64`, `seq = 2048`.

(a) How many total multiply-accumulate (MAC) operations does `Q · Kᵀ` require?

(b) The output `scores[i][j] = Q[i] · K[j]`. If you naively assign one thread per dot product (64 MACs each), how many threads do you need? What CUDA launch configuration would you use?

(c) Why is the naive single-thread-per-dot-product approach suboptimal? What memory access problem does it create, and what is the standard CUDA solution?

(d) After computing scores, you divide by `√d_k = √64 = 8`. Derive geometrically why this scaling is necessary — show what happens to the dot product variance as d_k grows.

---

**Solution:**

**(a) Total MACs for Q · Kᵀ:**

The result matrix is `[seq × seq] = [2048 × 2048]`. Each entry requires a dot product of length `d_k = 64`:

```
Total MACs = seq × seq × d_k
           = 2048 × 2048 × 64
           = 268,435,456
           ≈ 268 million multiply-accumulates
```

Per attention head. LLaMA-7B: 32 heads × 32 layers → **~274 billion MACs** per forward pass in attention alone.

**(b) Thread count and launch configuration:**

```
Number of output entries = seq × seq = 2048 × 2048 = 4,194,304

Naive: 1 thread per entry → 4,194,304 threads total

CUDA launch:
  THREADS_PER_BLOCK = 256 (or 512 — must be power of 2)
  BLOCKS = ceil(4,194,304 / 256) = 16,384

  dim3 grid(128, 128);   // 128×128 = 16,384 blocks (2D for matrix indexing)
  dim3 block(16, 16);    // 16×16 = 256 threads per block

  scores_naive<<<grid, block>>>(Q, K, scores, seq, d_k);
```

Each thread loops 64 times to compute one dot product:
```cpp
__global__ void scores_naive(const float* Q, const float* K,
                              float* scores, int seq, int dk) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key index
    if (i >= seq || j >= seq) return;
    float acc = 0.0f;
    for (int k = 0; k < dk; ++k) {
        acc += Q[i * dk + k] * K[j * dk + k];
    }
    scores[i * seq + j] = acc;
}
```

**(c) Memory access problem and solution:**

**The problem with Q access:**
Thread (i, j) accesses `Q[i*dk + 0], Q[i*dk + 1], ..., Q[i*dk + 63]`.
Thread (i, j+1) accesses the **same 64 elements** of Q.
All 2048 threads in row i read the same Q row — 2048 redundant global reads of 256 bytes each.

**The problem with K access:**
Thread (i, j) accesses `K[j*dk + k]` — strided by d_k.
For column-indexed threads in the same warp: consecutive threads access rows j, j+1, j+2... of K with stride d_k=64.
Warp access pattern: Thread 0 → K[j·64], Thread 1 → K[(j+1)·64], ... addresses are 256 bytes apart → **uncoalesced**.

**The standard solution — Tiled GEMM with shared memory:**

```cpp
#define TILE 16

__global__ void scores_tiled(const float* Q, const float* K,
                              float* scores, int seq, int dk) {
    __shared__ float Qs[TILE][TILE];   // tile of Q in shared memory
    __shared__ float Ks[TILE][TILE];   // tile of K in shared memory

    int row = blockIdx.y * TILE + threadIdx.y;  // output row (query)
    int col = blockIdx.x * TILE + threadIdx.x;  // output col (key)
    float acc = 0.0f;

    // Slide tile across dk dimension
    for (int t = 0; t < (dk + TILE - 1) / TILE; ++t) {
        // Coalesced load: consecutive threads → consecutive addresses
        int q_col = t * TILE + threadIdx.x;
        int k_col = t * TILE + threadIdx.y;

        Qs[threadIdx.y][threadIdx.x] = (row < seq && q_col < dk)
                                        ? Q[row * dk + q_col] : 0.0f;
        Ks[threadIdx.x][threadIdx.y] = (col < seq && k_col < dk)
                                        ? K[col * dk + k_col] : 0.0f;
        __syncthreads();

        // Compute partial dot products from shared memory (fast)
        for (int k = 0; k < TILE; ++k) {
            acc += Qs[threadIdx.y][k] * Ks[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < seq && col < seq) {
        scores[row * seq + col] = acc;
    }
}
```

**Memory traffic reduction:**
```
Naive:  Each of TILE×TILE = 256 threads reads TILE floats from global memory
        = 256 × 16 = 4,096 global reads per tile
Tiled:  TILE×TILE floats loaded ONCE into shared, each reused TILE=16 times
        = (16 + 16) global reads per tile
Reduction: 4,096 / 32 = ~128× fewer global memory transactions
```

**(d) Geometric derivation of the √d_k scaling:**

Assume Q and K components are drawn from a standard normal distribution:
```
Qᵢₖ ~ N(0, 1)   (zero mean, unit variance)
Kⱼₖ ~ N(0, 1)
```

For a single product `Qᵢₖ · Kⱼₖ`:
```
E[Qᵢₖ Kⱼₖ] = E[Qᵢₖ] · E[Kⱼₖ] = 0 · 0 = 0      (independent, zero mean)
Var[Qᵢₖ Kⱼₖ] = Var[Qᵢₖ] · Var[Kⱼₖ] = 1 · 1 = 1  (product of independent vars)
```

The dot product is the sum of d_k such independent products:
```
Q[i] · K[j] = Σₖ₌₁^{d_k} Qᵢₖ Kⱼₖ

E[Q[i] · K[j]] = 0

Var[Q[i] · K[j]] = Σₖ Var[Qᵢₖ Kⱼₖ] = d_k × 1 = d_k

Std[Q[i] · K[j]] = √d_k
```

**Without scaling:** At `d_k = 64`, attention scores have standard deviation **8**. The softmax:
```
softmax(scores)ⱼ = exp(scores[j]) / Σₖ exp(scores[k])
```
With scores ≈ 8× larger than expected, the maximum score dominates exponentially — softmax saturates to approximately a one-hot vector. Gradient: `∂softmax/∂x ≈ 0`. **Training fails.**

**With scaling:**
```
scaled_scores = Q[i] · K[j] / √d_k

Var[scaled_scores] = Var[Q[i]·K[j]] / d_k = d_k / d_k = 1
Std[scaled_scores] = 1
```

Now softmax operates in its sensitive region (scores ~ unit variance), gradients flow, training is stable regardless of `d_k`. This is not a heuristic — it is a direct consequence of the statistics of dot products in high-dimensional spaces. Every time you increase d_k, the dot product grows in variance proportionally, and `1/√d_k` exactly corrects for this.

---

*Assad's 52-Week NVIDIA Curriculum · Week 1: Linear Algebra I · Phase 1: Math & Systems*
*Saved: mathematics | File: week-01-linear-algebra-i-mathematics.md*
*Store at: `nvidia-curriculum/week-01-linear-algebra-i/week-01-linear-algebra-i-mathematics.md`*
