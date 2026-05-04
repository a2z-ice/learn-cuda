# Milestone: Environment Ready
> Week 01 | Phase 1: Math & Systems | Assad's NVIDIA Curriculum

---

## Status

| Item | Status |
|---|---|
| Milestone name | Environment Ready |
| Week | 01 of 52 |
| Target date | End of Week 1 (Friday) |
| Achieved | [ ] Fill in date when complete |

---

## What This Milestone Means

"Environment Ready" means end-to-end GPU compute has been demonstrated:

1. Code written locally
2. Compiled on a cloud GPU instance (RunPod)
3. Executed by real CUDA cores on real GPU hardware
4. Output verified correct
5. Screenshot committed to GitHub as permanent proof

This is not "I have an account." This is "I ran code on a GPU and can prove it."

---

## Verification Checklist

Complete every item. Screenshot the terminal showing all outputs simultaneously.

- [ ] `nvcc --version` → shows CUDA 12.x (or 11.x minimum)
- [ ] `nvidia-smi` → shows GPU model, driver version, VRAM
- [ ] `./hello_cuda` → prints results, all 1024 values = 2.000000
- [ ] Terminal hostname is RunPod instance (not localhost)
- [ ] Screenshot saved as `milestone-env-ready.png` in this folder
- [ ] Both files committed to GitHub

---

## How to Achieve This Milestone

### Step 1 — Provision RunPod Instance

```bash
# Go to runpod.io → Pods → + Deploy
# Select: GPU Pod
# GPU: RTX 3080 or RTX 4000 Ada (spot instance for cost)
# Template: RunPod PyTorch 2.x  (includes CUDA 12.x + nvcc)
# Storage: 10GB minimum
# Click Deploy
```

### Step 2 — SSH In

```bash
# From RunPod dashboard → Connect → SSH
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

### Step 3 — Verify Environment

```bash
nvcc --version       # Should show: Cuda compilation tools, release 12.x
nvidia-smi           # Should show your GPU, driver, VRAM
g++ --version        # Should show: g++ (Ubuntu ...) 11.x or later
```

### Step 4 — Clone Your Repo and Compile

```bash
git clone https://github.com/<your-username>/nvidia-curriculum.git
cd nvidia-curriculum/week-01-linear-algebra-i/cuda/
nvcc -o hello_cuda hello_cuda.cu
./hello_cuda
```

### Step 5 — Screenshot

Take a screenshot showing all of the following in one terminal view:
- The `nvcc --version` output
- The `nvidia-smi` output (or at minimum the GPU name line)
- The `./hello_cuda` output showing results and GPU info
- The hostname (confirms you're on RunPod, not local)

Save as: `milestone/milestone-env-ready.png`

### Step 6 — Compile C++ Programs (Bonus)

```bash
cd ../cpp/
make run
```

All three programs should compile and produce correct output.

---

## Fill In When Complete

```
Date achieved     : _______________
RunPod GPU model  : _______________
CUDA version      : _______________
Driver version    : _______________
VRAM              : _______________ GB
hello_cuda result : All 1024 values = 2.000000  [ ✓ / ✗ ]
Screenshot file   : milestone-env-ready.png      [ ✓ / ✗ ]
Committed to Git  : _______________
```

---

## Why This Goes in Your Portfolio

A NVIDIA recruiter or engineer reading your GitHub will look for evidence that you don't just read about GPU programming — you do it. This milestone document + screenshot is that evidence for Week 1.

It also establishes the baseline. Every week from here uses this same RunPod environment. Getting it right in Week 1 means no setup friction for the next 51 weeks.

---

*Assad's 52-Week NVIDIA Curriculum · Week 1 of 52 · Milestone: Environment Ready*
