#!/usr/bin/env python3
"""Build the HGA-1 data artifacts from the Uzbek Wikipedia parquet.

Produces, under --out:
  train.txt val.txt test.txt          cleaned, one document per line
  word_vocab.json                     <pad> <unk> <bos> <eos> + 16k words
  words_{train,val,test}.npy          int32, documents joined with <eos>
  char_features.npy                   float32 [V, 2*512] FHRR char encoding
  char_vocab.json                     char -> id map used by the encoder
  bpe.json bpe_{train,val,test}.npy   byte-level BPE, uint16
  meta.json                           byte and unit counts per split
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"[a-z\u02bb\u02bc'\u2018\u2019]+|[0-9]+|[^\sa-z0-9\u02bb\u02bc'\u2018\u2019]", re.I)
APOS = str.maketrans({"\u02bb": "'", "\u02bc": "'", "\u2018": "'", "\u2019": "'"})


def clean(text: str) -> str:
    text = text.translate(APOS).lower()
    text = re.sub(r"<[^>]{0,40}>", " ", text)          # wiki tags / refs
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]{0,80}\}\}", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    return WORD_RE.findall(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/home/tony/vsa-ghrr/data/uzwiki.parquet")
    ap.add_argument("--out", default="/home/tony/vsa-ghrr/data")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--char-dim", type=int, default=512)
    ap.add_argument("--max-word-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("[1/6] reading parquet", flush=True)
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet, columns=["text"])
    texts = table.column("text").to_pylist()
    print(f"      {len(texts)} documents ({time.time()-t0:.0f}s)", flush=True)

    print("[2/6] cleaning", flush=True)
    docs = []
    for t in texts:
        c = clean(t or "")
        if len(c) >= args.min_chars:
            docs.append(c)
        if args.max_docs and len(docs) >= args.max_docs:
            break
    print(f"      kept {len(docs)} docs, {sum(len(d) for d in docs)/1e6:.1f}M chars", flush=True)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(docs)
    n = len(docs)
    n_val = max(1, int(n * 0.01))
    n_test = max(1, int(n * 0.01))
    splits = {
        "test": docs[:n_test],
        "val": docs[n_test:n_test + n_val],
        "train": docs[n_test + n_val:],
    }
    for name, ds in splits.items():
        with open(os.path.join(args.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ds) + "\n")
        print(f"      {name}: {len(ds)} docs", flush=True)

    print("[3/6] word vocabulary", flush=True)
    counter = Counter()
    for d in splits["train"]:
        counter.update(tokenize(d))
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    itos = specials + [w for w, _ in counter.most_common(args.vocab_size - len(specials))]
    stoi = {w: i for i, w in enumerate(itos)}
    with open(os.path.join(args.out, "word_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"itos": itos, "stoi": stoi}, f, ensure_ascii=False)
    total_tokens = sum(counter.values())
    coverage = sum(counter[w] for w in itos[len(specials):]) / max(1, total_tokens)
    print(f"      vocab {len(itos)}, train tokens {total_tokens/1e6:.1f}M, "
          f"coverage {coverage*100:.2f}%", flush=True)

    print("[4/6] encoding word ids", flush=True)
    eos, unk = stoi["<eos>"], stoi["<unk>"]
    units = {}
    for name, ds in splits.items():
        out = []
        for d in ds:
            out.extend(stoi.get(w, unk) for w in tokenize(d))
            out.append(eos)
        arr = np.asarray(out, dtype=np.int32)
        np.save(os.path.join(args.out, f"words_{name}.npy"), arr)
        units[name] = int(arr.size)
        print(f"      {name}: {arr.size/1e6:.2f}M units", flush=True)

    print("[5/6] FHRR char features", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import importlib
    core = importlib.import_module("gpuvsa.vsacore_py")

    chars = Counter()
    for d in splits["train"][:20000]:
        chars.update(d)
    char_itos = ["<pad>", "<unk>"] + [c for c, _ in chars.most_common(46)]
    char_stoi = {c: i for i, c in enumerate(char_itos)}
    enc = core.CharEncoder(len(char_itos), args.char_dim, args.max_word_len, args.seed)
    feats = np.zeros((len(itos), args.char_dim * 2), dtype=np.float32)
    for i, w in enumerate(itos):
        ids = [char_stoi.get(c, 1) for c in w[: args.max_word_len]]
        if ids:
            z = np.asarray(enc.encode(ids), dtype=np.complex64)
            feats[i, 0::2] = z.real
            feats[i, 1::2] = z.imag
    np.save(os.path.join(args.out, "char_features.npy"), feats)
    with open(os.path.join(args.out, "char_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"itos": char_itos, "stoi": char_stoi}, f, ensure_ascii=False)
    print(f"      char vocab {len(char_itos)}, features {feats.shape}", flush=True)

    print("[6/6] BPE tokenizer", flush=True)
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 3)]
    )
    trainer = trainers.BpeTrainer(
        vocab_size=16000, min_frequency=2, show_progress=False,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([os.path.join(args.out, "train.txt")], trainer)
    tok.save(os.path.join(args.out, "bpe.json"))
    eos_bpe = tok.token_to_id("<eos>")
    bpe_units = {}
    for name, ds in splits.items():
        ids = []
        for d in ds:
            ids.extend(tok.encode(d, add_special_tokens=False).ids)
            ids.append(eos_bpe)
        arr = np.asarray(ids, dtype=np.uint16)
        np.save(os.path.join(args.out, f"bpe_{name}.npy"), arr)
        bpe_units[name] = int(arr.size)
        print(f"      {name}: {arr.size/1e6:.2f}M bpe units", flush=True)

    meta = {
        "bytes": {k: os.path.getsize(os.path.join(args.out, f"{k}.txt")) for k in splits},
        "units": units,
        "bpe_units": bpe_units,
        "word_vocab": len(itos),
        "bpe_vocab": tok.get_vocab_size(),
        "docs": {k: len(v) for k, v in splits.items()},
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
