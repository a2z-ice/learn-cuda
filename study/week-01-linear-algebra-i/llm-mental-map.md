# LLM Mental Map — Token to Token
> Week 01 | LLM Theory Track | Assad's NVIDIA Curriculum

---

## The Complete Pipeline (ASCII)

```
  INPUT TEXT
  "The cat sat"
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  TOKENIZER  (not a neural network — a lookup table)         │
│                                                             │
│  "The" → 464   "cat" → 3797   "sat" → 3332                │
│                                                             │
│  Output: [464, 3797, 3332]     shape: [seq_len]            │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER  (learned lookup table)                    │
│                                                             │
│  Each token ID → row of Embedding Matrix                    │
│  Matrix shape: [vocab_size=50257, d_model=768]              │
│                                                             │
│  464  → [0.21, -0.44, 0.87, ..., 0.13]   ← 768 floats    │
│  3797 → [0.05,  0.91, -0.22, ..., 0.67]  ← 768 floats    │
│  3332 → [-0.33, 0.12, 0.55, ..., -0.41]  ← 768 floats    │
│                                                             │
│  Output tensor shape: [3, 768]  (seq_len=3, d_model=768)  │
└─────────────────────────────────────────────────────────────┘
       │
       │  + Positional Encoding  [3, 768]  (adds position info)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCK × 12  (GPT-2 small has 12 layers)       │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MULTI-HEAD SELF-ATTENTION (12 heads, d_head=64)      │  │
│  │                                                       │  │
│  │  Q = x · Wq   [3, 768] × [768, 768] = [3, 768]      │  │
│  │  K = x · Wk   [3, 768] × [768, 768] = [3, 768]      │  │
│  │  V = x · Wv   [3, 768] × [768, 768] = [3, 768]      │  │
│  │                                                       │  │
│  │  scores = QKᵀ / √64    shape: [3, 3]  per head      │  │
│  │  ← This is a matrix of DOT PRODUCTS ←                │  │
│  │  ← Measures: how much does token i attend to j? ←    │  │
│  │                                                       │  │
│  │  weights = softmax(scores)   [3, 3]                  │  │
│  │  out = weights · V           [3, 64]  per head       │  │
│  │  concat all heads            [3, 768]                │  │
│  │  project: out · Wo           [3, 768]                │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                  residual connection (+x)                   │
│                  layer normalization                        │
│                         │                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  FEED-FORWARD NETWORK (per token, independently)      │  │
│  │                                                       │  │
│  │  h = ReLU(x · W1 + b1)  [3, 768] → [3, 3072]       │  │
│  │  out = h · W2 + b2       [3, 3072] → [3, 768]       │  │
│  │                                                       │  │
│  │  4× expansion then back to d_model                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                  residual connection (+x)                   │
│                  layer normalization                        │
└─────────────────────────────────────────────────────────────┘
       │
       │  (repeat × 12 for GPT-2 small)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  LANGUAGE MODEL HEAD  (final linear projection)             │
│                                                             │
│  last hidden state [3, 768]                                │
│  × W_lm_head [768, 50257]                                  │
│  = logits [3, 50257]                                       │
│                                                             │
│  We care about the LAST token's logits: [50257]            │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  SOFTMAX  →  PROBABILITY DISTRIBUTION                       │
│                                                             │
│  softmax([logits]) → [p₁, p₂, ..., p₅₀₂₅₇]               │
│  where Σ pᵢ = 1.0                                         │
│                                                             │
│  Highest probability token: "on" (ID 319) → p = 0.34      │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
  NEXT TOKEN: "on"
  New input: "The cat sat on"
  → Repeat from tokenizer
```

---

## Tensor Shape Summary Table (GPT-2 Small)

| Stage | Input Shape | Operation | Output Shape | Key Matrix |
|---|---|---|---|---|
| Tokenizer | raw text | BPE lookup | `[seq_len]` | vocab dict |
| Embedding | `[seq_len]` | index lookup | `[seq_len, 768]` | `[50257, 768]` |
| Pos. Encoding | `[seq_len, 768]` | addition | `[seq_len, 768]` | `[2048, 768]` |
| Attn Q/K/V | `[seq_len, 768]` | matmul | `[seq_len, 768]` each | `[768, 768]` × 3 |
| Attn scores | `[seq_len, 768]` | `QKᵀ/√64` | `[seq_len, seq_len]` | — |
| Attn output | `[seq_len, seq_len]` | `softmax·V` | `[seq_len, 768]` | `[768, 768]` |
| FFN up | `[seq_len, 768]` | matmul+ReLU | `[seq_len, 3072]` | `[768, 3072]` |
| FFN down | `[seq_len, 3072]` | matmul | `[seq_len, 768]` | `[3072, 768]` |
| LM head | `[seq_len, 768]` | matmul | `[seq_len, 50257]` | `[768, 50257]` |
| Softmax | `[50257]` | softmax | `[50257]` | — |

---

## Parameter Count Breakdown (GPT-2 Small)

| Component | Formula | Count |
|---|---|---|
| Token embedding | 50257 × 768 | 38.6M |
| Positional embedding | 1024 × 768 | 0.8M |
| Per-layer attn Q,K,V,O | 4 × (768 × 768) × 12 | 28.3M |
| Per-layer FFN W1, W2 | (768×3072 + 3072×768) × 12 | 56.6M |
| Layer norms + biases | small | ~0.1M |
| **Total** | | **~124M** |

---

## Key Concepts Annotated

### Why dot product is the core operation of attention

```
scores[i][j] = Q[i] · K[j]

This answers: "How similar is what token i is looking for (Q)
               to what token j is offering (K)?"

High dot product = vectors point same direction = high attention weight
Zero dot product = orthogonal = token i ignores token j
```

This is Week 1 math (dot product) directly inside Week 52 architecture (LLM attention). The chain is unbroken.

### What "parameters" physically are

Parameters are floating-point numbers stored in weight matrices on GPU global memory (VRAM). During inference, the forward pass is a series of matrix multiplications (`cudaGemm` calls under the hood). During training, gradients (also vectors/matrices) flow backward through these same matrices to update the weights.

When someone says "GPT-4 has ~1.8 trillion parameters," they mean there are 1.8 trillion floats living in GPU memory — and every inference involves computing dot products across all of them.

### The GPU parallelism entry point

```
FFN step:  output[i] = ReLU(input · W1[i])    for all i in [0, 3072)

Each output neuron is independent of every other.
→ 3072 independent dot products.
→ Run all 3072 in parallel on the GPU.
→ This is why GPUs with thousands of cores exist.
```

---

## Mental Model — One Sentence Per Stage

| Stage | One Sentence |
|---|---|
| Tokenizer | Chops text into integer IDs from a fixed vocabulary |
| Embedding | Looks up a learned vector for each token ID |
| Attention | Every token dot-products with every other token to decide what to focus on |
| FFN | Each token's representation gets further transformed by two linear layers |
| LM Head | Projects the final vector into a probability score over all possible next tokens |
| Sampling | Picks the next token; appends it; repeats from the top |

---

*Assad's 52-Week NVIDIA Curriculum · Week 1: Linear Algebra I · LLM Theory Track*
