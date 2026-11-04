#!/usr/bin/env python3
"""End-to-end retrofit experiment.

  baseline perplexity -> swap attention for GHRR attention -> verify the swap is
  lossless at init -> LoRA fine-tune -> perplexity again.

Every number is written to a JSON report next to the console log.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import torch  # noqa: E402

from gpuvsa.attention import install_ghrr  # noqa: E402
from gpuvsa.retrofit import (  # noqa: E402
    count_params,
    encode_corpus,
    finetune_lora,
    load_model,
    perplexity,
    save_report,
)


def get_corpus(split: str, limit: int = 400):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t.strip()]
    return texts[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="out/run.json")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--diagonality", type=float, default=0.5)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-gen", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-retrofit", action="store_true",
                    help="LoRA control on the untouched model")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    log = lambda *a: print(*a, flush=True)  # noqa: E731
    report = {"tag": args.tag, "model": args.model, "args": vars(args), "stages": {}}

    log(f"[1/5] loading {args.model} on {args.device}")
    t0 = time.time()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model, tok = load_model(args.model, dtype=dtype, device=args.device)
    total_p, _ = count_params(model)
    report["total_params"] = total_p
    log(f"      {total_p/1e6:.1f}M params, {time.time()-t0:.1f}s")

    log("[2/5] baseline perplexity (WikiText-2 test)")
    ids = encode_corpus(tok, get_corpus("test"))
    ppl0, ntok, secs = perplexity(model, ids, max_tokens=args.max_tokens, device=args.device,
                                  log=log)
    report["stages"]["baseline"] = {"ppl": ppl0, "tokens": ntok, "seconds": secs}
    log(f"      baseline ppl = {ppl0:.4f} over {ntok} tokens in {secs:.0f}s")

    if args.no_retrofit:
        binder = None
        report["binder_params"] = 0
        log("[3/5] control run: attention left untouched")
    else:
        log(f"[3/5] installing GHRR attention (block_size={args.block_size}, "
            f"diagonality={args.diagonality})")
        binder = install_ghrr(model, block_size=args.block_size, diagonality=args.diagonality)
        report["binder_params"] = sum(p.numel() for p in binder.parameters())

        log("[4/5] perplexity right after the swap (must match baseline)")
        ppl1, ntok1, secs1 = perplexity(model, ids, max_tokens=args.max_tokens,
                                        device=args.device, log=log)
        report["stages"]["retrofit_no_train"] = {
            "ppl": ppl1, "tokens": ntok1, "seconds": secs1, "delta_vs_baseline": ppl1 - ppl0,
        }
        log(f"      retrofit ppl = {ppl1:.4f} (delta {ppl1-ppl0:+.5f})")

    if not args.skip_train:
        log(f"[5/5] LoRA fine-tune, {args.steps} steps")
        train_ids = encode_corpus(tok, get_corpus("train", limit=2000))
        model, losses = finetune_lora(
            model, train_ids, binder, steps=args.steps,
            lr_lora=args.lr_lora, lr_gen=args.lr_gen, device=args.device, log=log,
        )
        report["stages"]["train"] = {
            "steps": args.steps,
            "first_loss": losses[0],
            "last_loss": sum(losses[-10:]) / len(losses[-10:]),
        }
        model.eval()
        ppl2, ntok2, secs2 = perplexity(model, ids, max_tokens=args.max_tokens,
                                        device=args.device, log=log)
        report["stages"]["after_lora"] = {"ppl": ppl2, "tokens": ntok2, "seconds": secs2,
                                          "delta_vs_baseline": ppl2 - ppl0}
        log(f"      after LoRA ppl = {ppl2:.4f} (delta {ppl2-ppl0:+.5f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_report(args.out, report)
    log(f"report -> {args.out}")
    print(json.dumps({k: v for k, v in report["stages"].items()}, indent=2))


if __name__ == "__main__":
    main()
