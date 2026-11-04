# vsa-ghrr — GHRR / HeRR kernels + GHRR attention retrofit on DGX Spark

This repository contains the working implementation behind the "1B model → GHRR"
plan: a C++ VSA kernel library (FHRR baseline, GHRR, HeRR), a Python binding, a
PyTorch attention module that binds token positions with block-unitary GHRR
matrices, and an end-to-end retrofit + LoRA experiment that is **numerically
verifiable**: with `diagonality=1.0` the retrofitted model reproduces the
original model's logits, because the positional binding is exactly RoPE.

## Layout

```
include/vsa/        header-only C++ core
  fhrr.hpp          FHRR baseline (elementwise complex product)
  ghrr.hpp          GHRR: D components of m x m unitaries, bind/unbind/left_unbind
  herr.hpp          HeRR: Laurent polynomials + mean-and-peel decoder
  common.hpp        random unitaries, small matrix helpers
src/bench.cpp       capacity / noise / algebra benchmarks (JSONL out)
tests/test_vsa.cpp  39 assertions on the algebra
python/
  bindings.cpp      pybind11 module (vsacore_py)
  gpuvsa/attention.py   GHRRPositionBinder + GHRRAttention + install_ghrr
  gpuvsa/retrofit.py    perplexity, LoRA fine-tune, report helpers
scripts/run_pipeline.py  the full experiment
tests/test_attention.py  equivalence / causality / gradient tests (CPU)
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vsa-tests
./build/vsa-bench capacity --trials=8 > out/capacity.jsonl
./build/vsa-bench herr --trials=5     > out/herr.jsonl
./build/vsa-bench noise               > out/noise.jsonl
```

Python bindings (needs `pybind11`):

```bash
cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release -DVSA_PYTHON=ON
cmake --build build-py -j
python -c "import sys; sys.path.insert(0,'python'); from gpuvsa import GHRR; print(GHRR.random_unitary(8,3,0.5,1).array.shape)"
```

## The attention module

`GHRRPositionBinder` splits a head vector into `D = head_dim / m` blocks of `m`
consecutive dimensions and binds each block with `E_b(pos) = expm(pos * G_b)`,
`G_b` real skew-symmetric. Because `E(a)E(b) = E(a+b)`, scores depend only on
the relative position. With `m=2` and `G_b = theta_b * [[0,-1],[1,0]]` this is
bit-for-bit RoPE after the dimension permutation Llama/Qwen use.

`diagonality < 1` adds a learnable skew-symmetric generator to a fraction of the
blocks, initialised at zero. The model therefore starts exactly at RoPE (so the
retrofit is lossless) and can learn block-internal, non-commutative mixing.

## Retrofit experiment

```bash
python scripts/run_pipeline.py --model Qwen/Qwen2.5-0.5B \
    --diagonality 0.5 --steps 150 --max-tokens 40000 \
    --out out/run_diag5.json
```

Stages: baseline perplexity → swap attention → perplexity (must match) → LoRA →
perplexity. See `reports/RETROFIT_REPORT.md` for measured numbers.

## HGA-1 (word-level FHRR + GHRR model)

A second, self-contained experiment lives in `python/hga/`: a word-level LM that
composes words from characters with the FHRR encoder, projects them into
`d_model=384`, and runs six GHRR-attention blocks (m=4, `head_dim=64`, D=16).

```bash
python python/hga/data.py --parquet data/uzwiki.parquet --out data
python python/hga/train.py --model hga --data data --ckpt runs/hga.pt --steps 3000 \
    --batch 32 --seq 512 --lr 3e-4 --warmup 200 --device cuda --dtype bf16
python python/hga/eval.py  --model hga --ckpt runs/hga.pt --data data --split test
```

Measured on Uzbek Wikipedia (359 MB train, 3.96 MB test), equal compute:

| model | params | steps | wall | BPB | ppl |
|---|---|---|---|---|---|
| HGA-1 (words, GHRR) | 17.50M | 3000 | 1073 s | **0.8487** | 37.24 (word) |
| Baseline (16k BPE) | 17.18M | 4270 | 1088 s | 1.2605 | 42.77 (token) |

Both models process ~278 kB of text per second, so the runs are compute- and
data-matched. Details and caveats: `reports/HGA1_REPORT.md`.
