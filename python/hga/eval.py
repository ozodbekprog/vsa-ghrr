#!/usr/bin/env python
"""Evaluation harness: BPB / perplexity / semantic probe for HGA-1 vs baseline.

Follows docs/HGA1_CONTRACT.md:
  bpb = CE_total / (n_bytes * ln 2)

The split is scored with stride seq-1 windows, so every unit after the first
is predicted exactly once with up to seq-1 units of context.  When only part
of the split is evaluated (--max-units N), n_bytes is scaled by the fraction
of units consumed: bytes_eff = meta.bytes[split] * (N / meta.units[split]).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

DATA_FILE = {"hga": "words_{split}.npy", "baseline": "bpe_{split}.npy"}
SEMANTIC_WORDS = ["olma", "meva", "stol", "daraxt", "non", "suv", "kitob", "maktab"]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_char_features(data_dir: Path):
    path = data_dir / "char_features.npy"
    if not path.exists():
        return None
    return torch.from_numpy(np.ascontiguousarray(np.load(path), dtype=np.float32))


def build_model_from_args(model_args: dict, char_features=None) -> torch.nn.Module:
    arch = model_args["arch"]
    if arch == "hga":
        cls = importlib.import_module("hga.model_hga").HGA1Model
        kwargs = dict(
            vocab_size=model_args["vocab_size"],
            d_model=model_args["d_model"],
            n_blocks=model_args["n_blocks"],
            n_heads=model_args["n_heads"],
            block_size=model_args.get("block_size", 4),
            ffn_dim=model_args["ffn_dim"],
            dropout=model_args.get("dropout", 0.0),
        )
        if model_args.get("has_char_features") and char_features is not None:
            if "char_features" not in inspect.signature(cls.__init__).parameters:
                raise TypeError(
                    "checkpoint expects char_features but HGA1Model does not "
                    "accept it")
            kwargs["char_features"] = char_features
    else:
        cls = importlib.import_module("hga.model_baseline").BaselineLM
        kwargs = dict(
            vocab_size=model_args["vocab_size"],
            d_model=model_args["d_model"],
            n_blocks=model_args["n_blocks"],
            n_heads=model_args["n_heads"],
            ffn_dim=model_args["ffn_dim"],
            dropout=model_args.get("dropout", 0.0),
            max_pos=model_args["max_pos"],
        )
    return cls(**kwargs)


def semantic_probe(model, data_dir: Path, model_kind: str) -> dict:
    """Contract semantic test on output_embeddings() rows."""
    if model_kind != "hga":
        note = ("skipped: baseline embeddings live in BPE token space; mapping "
                "the word list through the BPE tokenizer is out of scope")
        print(f"[semantic] {note}", flush=True)
        return {"status": "skipped", "reason": note}

    wv = data_dir / "word_vocab.json"
    if not wv.exists():
        note = "skipped: data/word_vocab.json missing"
        print(f"[semantic] {note}", flush=True)
        return {"status": "skipped", "reason": note}
    stoi = read_json(wv).get("stoi", {})
    missing = [w for w in SEMANTIC_WORDS if w not in stoi]
    if missing:
        note = f"skipped: words absent from vocab: {missing}"
        print(f"[semantic] {note}", flush=True)
        return {"status": "skipped", "reason": note}

    ids = [int(stoi[w]) for w in SEMANTIC_WORDS]
    with torch.no_grad():
        emb = model.output_embeddings().detach().float().cpu()
    if emb.ndim != 2 or max(ids) >= emb.shape[0]:
        note = "skipped: output_embeddings() rows unavailable"
        print(f"[semantic] {note}", flush=True)
        return {"status": "skipped", "reason": note}

    rows = emb[ids]
    rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx = {w: i for i, w in enumerate(SEMANTIC_WORDS)}

    def cos(a: str, b: str) -> float:
        return float(torch.dot(rows[idx[a]], rows[idx[b]]))

    c_fruit = cos("olma", "meva")
    c_tree = cos("olma", "daraxt")
    c_table = cos("olma", "stol")
    fruit_pass = c_fruit > c_table
    tree_pass = c_tree > c_table

    print(f"[semantic] cos(olma,meva)={c_fruit:.4f} vs cos(olma,stol)={c_table:.4f}"
          f" -> {'PASS' if fruit_pass else 'FAIL'}", flush=True)
    print(f"[semantic] cos(olma,daraxt)={c_tree:.4f} vs cos(olma,stol)={c_table:.4f}"
          f" -> {'PASS' if tree_pass else 'FAIL'}", flush=True)
    return {
        "status": "ok",
        "words": SEMANTIC_WORDS,
        "cosines": {"olma_meva": c_fruit, "olma_daraxt": c_tree,
                    "olma_stol": c_table},
        "fruit_closer_than_table": fruit_pass,
        "tree_closer_than_table": tree_pass,
        "pass": bool(fruit_pass and tree_pass),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["hga", "baseline"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu", flush=True)
        args.device = "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    if model_args["arch"] != args.model:
        raise ValueError(
            f"checkpoint arch={model_args['arch']} but --model {args.model}")
    train_args = ckpt.get("train_args") or {}
    seq = int(args.seq or train_args.get("seq") or 512)

    data_dir = Path(args.data)
    path = data_dir / DATA_FILE[args.model].format(split=args.split)
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    arr = np.load(path, mmap_mode="r").reshape(-1)
    total_units = int(arr.shape[0])
    eval_units = total_units if args.max_units is None else min(total_units,
                                                                args.max_units)
    if eval_units < 2:
        raise ValueError("need at least 2 units to evaluate")

    char_features = load_char_features(data_dir) if args.model == "hga" else None
    if model_args.get("has_char_features") and char_features is None:
        raise FileNotFoundError(
            "checkpoint expects char_features but data/char_features.npy is missing")
    model = build_model_from_args(model_args, char_features)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        print(f"[warn] strict load failed ({exc}); retrying strict=False", flush=True)
        model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    use_amp = args.device.startswith("cuda")
    autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
                else torch.autocast("cpu", enabled=False))

    starts = list(range(0, eval_units - 1, seq - 1))
    ce_sum = 0.0
    n_units = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        i = 0
        while i < len(starts):
            batch_starts = starts[i:i + args.batch]
            wins = [np.asarray(arr[s:min(s + seq, eval_units)]) for s in batch_starts]
            lengths = {len(w) for w in wins}
            if len(lengths) > 1:  # ragged tail: score the short window alone
                wins, batch_starts = wins[:1], batch_starts[:1]
            if len(wins[0]) < 2:
                i += 1
                continue
            x = torch.from_numpy(np.stack(wins)).to(args.device).long()
            with autocast:
                logits = model(x[:, :-1])
            logits = logits.reshape(-1, logits.shape[-1]).float()
            ce = F.cross_entropy(logits, x[:, 1:].reshape(-1), reduction="sum")
            ce_sum += float(ce)
            n_units += int(x.shape[0] * (x.shape[1] - 1))
            i += len(batch_starts)
    wall = time.perf_counter() - t0

    if n_units <= 0:
        raise RuntimeError("no units evaluated")
    ce_per_unit = ce_sum / n_units
    ppl = math.exp(ce_per_unit)

    meta_path = data_dir / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    bytes_full = meta.get("bytes", {}).get(args.split)
    # BPE splits must be scaled by the BPE unit count, not the word count.
    unit_key = "units" if args.model == "hga" else "bpe_units"
    units_meta = meta.get(unit_key, {}).get(args.split)
    if units_meta:
        fraction = eval_units / float(units_meta)
    else:
        fraction = eval_units / float(total_units)
    bytes_eff = None if bytes_full is None else float(bytes_full) * fraction
    bpb = None if not bytes_eff else ce_sum / (bytes_eff * math.log(2.0))

    semantic = semantic_probe(model, data_dir, args.model)

    result = {
        "model": args.model,
        "ckpt": str(args.ckpt),
        "split": args.split,
        "step": int(ckpt.get("step", -1)),
        "seq": seq,
        "units_in_split": total_units,
        "units_consumed": eval_units,
        "bytes_scaled": bytes_eff,
        "n_units": n_units,
        "ce_sum": ce_sum,
        "ce_per_unit": ce_per_unit,
        "ppl": ppl,
        "bpb": bpb,
        "wall_s": wall,
        "units_per_s": n_units / max(1e-9, wall),
        "semantic": semantic,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    sem = {"ok": "PASS" if semantic.get("pass") else "FAIL",
           "skipped": "SKIP"}.get(semantic.get("status"), semantic.get("status"))
    bpb_s = "n/a" if bpb is None else f"{bpb:.4f}"
    print(f"RESULT model={args.model} split={args.split} step={result['step']} "
          f"n_units={n_units} ce={ce_sum:.1f} ppl={ppl:.3f} bpb={bpb_s} "
          f"units_per_s={result['units_per_s']:.0f} wall_s={wall:.1f} "
          f"semantic={sem} out={out}", flush=True)


if __name__ == "__main__":
    main()
