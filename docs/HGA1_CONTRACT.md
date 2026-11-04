# HGA-1 build contract — read fully before writing code

Repo (macOS copy): `/Users/probox/Documents/dgx-ghrr`
Spark (test machine): `<spark-host>:/home/tony/vsa-ghrr`
Spark python: `/home/tony/vsa-venv/bin/python` (torch 2.9.1+cu130, GB10, transformers 4.46.3)
You write ONLY your assigned files. Never edit a file owned by another agent.

## Dimension decision (the spec is internally inconsistent)

The spec says `D_ghrr = 96 = heads soni` while also `n_heads=6`. 96 != 6, and
96 blocks x 4x4 does not fit `d_model=384`. Use the dimensionally consistent
choice everywhere:

```
d_model = 384, n_heads = 6, head_dim = 64, block_size m = 4
-> D (components per head) = head_dim / m = 16
-> GHRR coefficients per head = 16 * 4 * 4 = 256
```

`D_ghrr` is derived (`D = head_dim // m`). Keep it parametric.

## Data artifacts (produced by the data agent, under `data/`)

- `train.txt` / `val.txt` / `test.txt` — cleaned UTF-8, one document per line
- `word_vocab.json` — `{"itos": ["<pad>","<unk>","<bos>","<eos>", ...], "stoi": {...}}`
  ids: pad=0, unk=1, bos=2, eos=3, then words by frequency (vocab 16000)
- `words_{train,val,test}.npy` — int32, documents joined with `<eos>`
- `bpe.json` — a `tokenizers` BPE tokenizer trained on `train.txt`, vocab 16000
- `bpe_{train,val,test}.npy` — uint16, documents joined by eos id
- `meta.json` — `{"bytes": {"train": N, ...}, "units": {"train": N, ...}}`

## Model interfaces (must match exactly)

`python/hga/model_hga.py`

```python
class GHRRBlock(nn.Module):      # pre-LN transformer block
    ...

class HGA1Model(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_blocks=6, n_heads=6,
                 block_size=4, ffn_dim=1536, dropout=0.0)
    def forward(self, word_ids):  # [B,W] long -> logits [B,W,vocab]
    def input_embeddings(self, word_ids):   # [N, d_model]  (proj output)
    def output_embeddings(self):            # [vocab, d_model] (lm_head rows)
```

`python/hga/model_baseline.py`

```python
class BaselineLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_blocks=6, n_heads=6,
                 ffn_dim=1536, dropout=0.0, max_pos=1024)
    def forward(self, token_ids):  # [B,T] long -> logits [B,T,vocab]
    def input_embeddings(self, token_ids):  # [N, d_model]
    def output_embeddings(self):            # [vocab, d_model]
```

Both models must be ~equal size with **tied** embeddings (~17-19M params).
Baseline = learned token embedding + standard softmax attention (eager, causal
mask), same depth/width/FFN. It is the control for BPB.

## Position binding

Reuse the verified module `python/gpuvsa/attention.py::GHRRPositionBinder`.
It splits a head vector into blocks of `m` consecutive dims; with `m=4` it
folds two RoPE pairs per block. Bind **q and k only**; values stay unbound.
Use `diagonality=1.0`, generators frozen (measured: learnable extras do not
help at short budgets). Explain this in a comment.

## Character encoder (agent 1 owns)

`include/vsa/char_encoder.hpp` + binding in `python/bindings.cpp` + tests.

```cpp
struct CharEncoder {                 // FHRR random indexing
    CharEncoder(int vocab, int D, int max_len, uint64_t seed);
    std::vector<std::complex<float>> encode(const std::vector<int>& ids) const;
    void encode_batch(const int* ids, const int* lens, int n_words, int max_len,
                      std::complex<float>* out) const;  // out: n_words x D
};
```

`encode(w) = normalize( sum_i char_vec[c_i] * pos_vec[i] )`. Deterministic by
seed. Required tests: `encode("olma")` vs `encode("amlo")` cosine < 0.2;
self-cosine == 1; different lengths differ; determinism.

## Eval contract

- `bpb = CE_total / (n_bytes * ln 2)`, where `n_bytes` is the UTF-8 byte count
  of the evaluated text and `CE_total` is the summed cross-entropy.
- Semantic test uses `output_embeddings()` rows for
  `olma, meva, stol, daraxt, non, suv, kitob, maktab`, reporting
  `cos(olma,meva)` vs `cos(olma,stol)` and `cos(olma,daraxt)` vs `cos(olma,stol)`.
