#!/usr/bin/env python
"""Training harness for HGA-1 (word units) vs the BPE baseline.

Interfaces follow docs/HGA1_CONTRACT.md exactly:
  * HGA1Model / BaselineLM are imported from python/hga/model_{hga,baseline}.py
  * HGA consumes data/words_{split}.npy (int32, docs joined with <eos>)
  * baseline consumes data/bpe_{split}.npy (uint16, docs joined by eos id)
  * loss is plain cross-entropy of the shifted window (unit t given 0..t-1)

Sampling is seeded per (seed, step), so an interrupted run resumed with
``--resume`` replays exactly the same batches and LR schedule.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

DATA_FILE = {"hga": "words_{split}.npy", "baseline": "bpe_{split}.npy"}


# ---------------------------------------------------------------- utilities
def is_tiny() -> bool:
    return os.environ.get("HGA_TEST_TINY", "").lower() in {"1", "true", "yes"}


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_array(data_dir: Path, arch: str, split: str) -> np.ndarray:
    path = data_dir / DATA_FILE[arch].format(split=split)
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    arr = np.load(path, mmap_mode="r")
    return arr.reshape(-1)


def array_max(arr: np.ndarray, chunk: int = 4_000_000) -> int:
    """Max id without materialising a large memmap in RAM."""
    best = 0
    n = arr.shape[0]
    for lo in range(0, n, chunk):
        best = max(best, int(np.asarray(arr[lo:lo + chunk]).max(initial=0)))
    return best


def infer_vocab_size(data_dir: Path, arch: str, arr: np.ndarray) -> int:
    if arch == "hga":
        wv = data_dir / "word_vocab.json"
        if wv.exists():
            obj = read_json(wv)
            if "itos" in obj:
                return len(obj["itos"])
            if "stoi" in obj:
                return len(obj["stoi"])
    else:
        bpe = data_dir / "bpe.json"
        if bpe.exists():
            try:
                from tokenizers import Tokenizer  # optional, data agent owns it
                return int(Tokenizer.from_file(str(bpe)).get_vocab_size())
            except Exception:
                pass
    return array_max(arr) + 1


def load_char_features(data_dir: Path):
    path = data_dir / "char_features.npy"
    if not path.exists():
        return None
    feats = np.load(path)
    return torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))


def dims_for(seq: int) -> dict:
    if is_tiny():
        # Fast CPU/GPU smoke configuration (tests/test_harness.py).
        return dict(d_model=64, n_blocks=2, n_heads=2, ffn_dim=256)
    return dict(d_model=384, n_blocks=6, n_heads=6, ffn_dim=1536)


def build_model_from_args(model_args: dict, char_features=None) -> torch.nn.Module:
    """Instantiate the contract model from a plain dict (checkpoint friendly)."""
    arch = model_args["arch"]
    if arch == "hga":
        mod = importlib.import_module("hga.model_hga")
        cls = mod.HGA1Model
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
            params = inspect.signature(cls.__init__).parameters
            if "char_features" not in params:
                raise TypeError(
                    "data/char_features.npy exists but HGA1Model does not accept "
                    "char_features; refusing to silently drop it")
            kwargs["char_features"] = char_features
    else:
        mod = importlib.import_module("hga.model_baseline")
        cls = mod.BaselineLM
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


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    floor = base_lr * min_ratio
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, opt, scaler, step: int,
                    model_args: dict, train_args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "model_args": model_args,
        "train_args": train_args,
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["hga", "baseline"], required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/train.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/ckpt.pt")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu", flush=True)
        args.device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    out_path = Path(args.out)
    ckpt_path = Path(args.ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_arr = split_array(data_dir, args.model, "train")
    if train_arr.shape[0] <= args.seq + 1:
        raise ValueError(f"split too short ({train_arr.shape[0]}) for --seq {args.seq}")

    dims = dims_for(args.seq)
    char_features = load_char_features(data_dir) if args.model == "hga" else None
    if char_features is not None:
        cls = importlib.import_module("hga.model_hga").HGA1Model
        if "char_features" not in inspect.signature(cls.__init__).parameters:
            print("[warn] data/char_features.npy exists but HGA1Model has no "
                  "char_features argument; falling back to the model default",
                  flush=True)
            char_features = None

    start_step = 0
    ckpt = None
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_args = ckpt["model_args"]
        start_step = int(ckpt["step"])
        print(f"[resume] {ckpt_path} at step {start_step}", flush=True)
    else:
        if args.resume:
            print(f"[warn] --resume given but {ckpt_path} not found; starting fresh",
                  flush=True)
        vocab = infer_vocab_size(data_dir, args.model, train_arr)
        model_args = {
            "arch": args.model,
            "vocab_size": int(vocab),
            **dims,
            "block_size": 4,
            "dropout": 0.0,
            "max_pos": max(1024, args.seq + 1) if not is_tiny() else max(128, args.seq + 1),
            "has_char_features": char_features is not None,
            "tiny": is_tiny(),
        }

    if model_args.get("has_char_features") and char_features is None:
        raise FileNotFoundError(
            "checkpoint was trained with char_features but data/char_features.npy "
            "is not present")

    model = build_model_from_args(model_args, char_features).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])

    use_amp = args.device.startswith("cuda") and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.dtype == "fp16")
    if ckpt is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    autocast = (torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp
                else nullcontext())

    model.train()
    n_units = train_arr.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] arch={args.model} vocab={model_args['vocab_size']} "
          f"params={n_params/1e6:.2f}M units={n_units} seq={args.seq} "
          f"device={args.device} dtype={args.dtype} tiny={is_tiny()}", flush=True)

    train_args = {
        "model": args.model, "data": str(data_dir), "batch": args.batch,
        "seq": args.seq, "lr": args.lr, "warmup": args.warmup,
        "clip": args.clip, "seed": args.seed, "dtype": args.dtype,
        "steps": args.steps,
    }

    mode = "a" if (args.resume and out_path.exists()) else "w"
    log_fh = open(out_path, mode, encoding="utf-8")
    t0 = time.perf_counter()
    t_log = t0
    tokens_log = 0
    step = start_step
    last_loss = float("nan")
    stop_reason = "steps"

    def maybe_save(s: int) -> None:
        save_checkpoint(ckpt_path, model, opt, scaler, s, model_args, train_args)

    done = start_step
    try:
        for step in range(start_step, args.steps):
            if args.max_minutes is not None and \
                    (time.perf_counter() - t0) >= args.max_minutes * 60.0:
                stop_reason = "max_minutes"
                break

            lr = lr_at(step, args.lr, args.warmup, args.steps)
            for group in opt.param_groups:
                group["lr"] = lr

            rng = np.random.default_rng([args.seed, step])
            starts = rng.integers(0, n_units - args.seq, size=args.batch)
            windows = np.stack([np.asarray(train_arr[s:s + args.seq])
                                for s in starts]).astype(np.int64, copy=False)
            x = torch.from_numpy(windows).to(args.device)
            inp, tgt = x[:, :-1], x[:, 1:]

            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            if not math.isfinite(float(loss)):
                print(f"[warn] non-finite loss at step {step}; skipping update",
                      flush=True)
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            last_loss = float(loss.detach())
            done = step + 1
            tokens_log += args.batch * args.seq

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rec = {
                    "step": step + 1,
                    "loss": last_loss,
                    "lr": lr,
                    "tokens_per_s": tokens_log / max(1e-9, now - t_log),
                    "wall_s": now - t0,
                }
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                t_log, tokens_log = now, 0
                print(json.dumps(rec), flush=True)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                maybe_save(step + 1)
    finally:
        log_fh.close()

    # Always persist the last state, including max-minutes stops.
    maybe_save(done)
    wall = time.perf_counter() - t0
    units_per_s = (done - start_step) * args.batch * args.seq / max(1e-9, wall)
    print(f"SUMMARY model={args.model} steps={done} wall_s={wall:.1f} "
          f"units_per_s={units_per_s:.1f} loss={last_loss:.4f} "
          f"stop={stop_reason} ckpt={ckpt_path} log={out_path}", flush=True)


if __name__ == "__main__":
    main()
