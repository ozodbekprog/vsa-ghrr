"""Byte-level BPE for the BPB control arm.

Train a `tokenizers` byte-level BPE on a plain-text file (one document per line)
and save it as JSON:

    python python/hga/bpe.py --input data/train.txt --vocab-size 16000 \
        --out data/bpe.json

The run is idempotent: BPE training is deterministic for a fixed input file and
fixed hyper-parameters, and the vocabulary is written atomically (tmp + rename),
so re-running the same command reproduces byte-identical `bpe.json`.

`encode_file()` turns a text file into the uint16 token stream the trainer
consumes, appending `<eos>` after every document line.  numpy is imported lazily
so that this module can be imported (e.g. for `--help`) without numpy installed.
"""
from __future__ import annotations

import argparse
import os
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

__all__ = ["SPECIAL_TOKENS", "build_tokenizer", "train_bpe", "encode_file", "main"]


def build_tokenizer(vocab_size: int, split: str = "bytelevel", min_frequency: int = 2):
    """Construct an untrained byte-level BPE tokenizer + its trainer."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    if split == "whitespace":
        # Pre-split on whitespace, then byte-map each piece.
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif split == "bytelevel":
        # GPT-2 style regex split + byte mapping in one step.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    else:
        raise ValueError(f"unknown split mode: {split!r}")

    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    return tok, trainer


def train_bpe(input_path: str, out_path: str, vocab_size: int = 16000,
              split: str = "bytelevel", min_frequency: int = 2):
    """Train on `input_path`, save JSON to `out_path`, return the final vocab size."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    tok, trainer = build_tokenizer(vocab_size, split=split, min_frequency=min_frequency)
    tok.train([input_path], trainer)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "." + os.path.basename(out_path) + ".tmp")
    tok.save(tmp_path)
    os.replace(tmp_path, out_path)  # atomic: never leaves a half-written bpe.json
    return tok.get_vocab_size()


def encode_file(tok_path: str, text_path: str):
    """Tokenize `text_path` with the BPE at `tok_path`.

    Returns a uint16 numpy array: each non-empty document line is encoded
    (without added specials) followed by one `<eos>` id, matching the layout of
    the word-level `words_*.npy` artifacts.  numpy is imported lazily.
    """
    import numpy as np  # lazy: only needed by the encoding path

    tok = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError(f"{tok_path} has no <eos> token; retrain with bpe.py")
    vocab = tok.get_vocab_size()
    if vocab > 65535:
        raise ValueError(f"vocab size {vocab} does not fit in uint16")

    docs = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                docs.append(line)

    encodings = tok.encode_batch(docs, add_special_tokens=False)
    total = sum(len(e.ids) for e in encodings) + len(encodings)
    out = np.empty(total, dtype=np.uint16)
    i = 0
    for enc in encodings:
        n = len(enc.ids)
        out[i:i + n] = enc.ids
        i += n
        out[i] = eos_id
        i += 1
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer (tokenizers) and save bpe.json.")
    p.add_argument("--input", required=True, help="plain UTF-8 text, one document per line")
    p.add_argument("--vocab-size", type=int, default=16000, help="target vocab size (default 16000)")
    p.add_argument("--out", default="data/bpe.json", help="output tokenizer JSON (default data/bpe.json)")
    p.add_argument("--split", choices=["bytelevel", "whitespace"], default="bytelevel",
                   help="pre-tokenizer split (default bytelevel, GPT-2 regex)")
    p.add_argument("--min-frequency", type=int, default=2,
                   help="minimum merge frequency (default 2); may cap the final vocab below --vocab-size")
    args = p.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    n_bytes = os.path.getsize(args.input)
    final = train_bpe(args.input, args.out, args.vocab_size,
                      split=args.split, min_frequency=args.min_frequency)
    print(f"input: {args.input} ({n_bytes} bytes)")
    print(f"requested vocab size: {args.vocab_size}")
    print(f"final vocab size: {final}")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
