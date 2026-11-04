# Measured results — index

All numbers below were produced on a DGX Spark (NVIDIA GB10, 121 GB unified,
Ubuntu 24.04 aarch64) with PyTorch 2.9.1+cu130. Raw artifacts live next to this
file: `runs/*.json[l]` (training logs and evals), `out/*.json[l]` (kernel
benchmarks and retrofit runs), full write-ups in `reports/`.

---

## 1. VSA kernels (C++ core)

Role–filler retrieval accuracy, 8–10 trials per cell. `out/capacity.jsonl`,
`out/herr.jsonl`, `out/noise.jsonl`.

| FHRR D | M=16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| 128 | 0.98 | 0.76 | 0.40 | 0.20 | 0.15 |
| 256 | 1.00 | 0.93 | 0.67 | 0.43 | 0.26 |
| 512 | 1.00 | 1.00 | 0.93 | 0.77 | 0.60 |
| 1024 | 1.00 | 1.00 | 1.00 | 0.97 | 0.92 |

GHRR at equal coefficient count matches FHRR (D=128,m=2 → 512 coeffs, M=64:
0.94 vs FHRR-512 0.93); its advantage is non-commutativity
(`sim(a⊙b, b⊙a) = 0.48` at m=3, diag=0), not capacity.

HeRR with sparse fillers reaches 96/96 pairs at 1.00 accuracy, but the memory
support grows to **4141 coefficients** for D=128 fillers — the "96 pairs in 128
complex numbers" claim did not reproduce.

Noise (FHRR, D=128, M=16): 0.50 at −6 dB → 0.97 at +6 dB.

---

## 2. GHRR attention retrofit (Qwen2.5-0.5B, WikiText-2)

40 512 test tokens. `out/run_*.json`.

| stage | perplexity |
|---|---|
| baseline | 15.7446 |
| after attention swap (no training) | **15.7453** (Δ +0.0008) |
| + 300 LoRA steps, original attention | 11.2681 |
| + 300 LoRA steps, GHRR frozen at RoPE | 11.2539 |
| + 300 LoRA steps, GHRR learnable blocks (lr 1e-3) | 11.3569 |
| + 300 LoRA steps, GHRR learnable blocks (lr 1e-2) | 11.7215 |

Verification: with `diagonality=1.0` the swapped model reproduces the original
logits to **1.5e-7** (11/11 tests in `tests/test_attention.py`).

---

## 3. HGA-1 vs BPE baseline (Uzbek Wikipedia)

Train 359 MB / 59.15M words / 84.08M BPE tokens; test 3.96 MB. Equal compute
(~1080 s each, ~299 MB of text each).

| | HGA-1 (words, GHRR) | Baseline (16k BPE) |
|---|---|---|
| params | 17.50M | 17.18M |
| steps | 3000 | 4270 |
| train wall | 1073 s | 1088 s |
| throughput | 45 804 words/s (278 kB/s) | 64 318 tokens/s (275 kB/s) |
| test ppl | 37.24 (word) | 42.77 (token) |
| **test BPB** | **0.8487** | 1.2605 |

**BPB advantage: 32.7 %.** `<unk>` is 15.0 % of test tokens but only 5.9 % of the
cross-entropy; excluding unk targets puts HGA-1 at ≈0.97 BPB, still ≈23 % ahead.

Semantic probe (output embeddings):
`cos(olma,meva)=0.5902`, `cos(olma,stol)=0.5443`, `cos(olma,daraxt)=0.5396`
→ "olma–meva" passes, "olma–daraxt" fails by 0.005.

`runs/eval_hga.json`, `runs/eval_base_4270.json`, `reports/HGA1_REPORT.md`.

---

## 4. Test suite

```
./build/vsa-tests              39 passed
./build/test-char-encoder      19 passed
tests/test_attention.py        11 passed
tests/test_model_hga.py        pass (incl. causality, grads, overfit)
tests/test_baseline.py         pass
tests/test_harness.py          pass (train→eval round trip)
```

## 5. Not in git (large)

`data/uzwiki.parquet` (210 MB), tokenised arrays (~500 MB), `runs/*.pt`
(263 MB HGA, 197 MB baseline). They stay on the Spark at
`~/vsa-ghrr/data` and `~/vsa-ghrr/runs`; the pipeline regenerates the data
with `python python/hga/data.py` and the checkpoints with `train.py`.
