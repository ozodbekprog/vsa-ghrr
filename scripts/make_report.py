#!/usr/bin/env python3
"""Turn the raw benchmark JSONL files into Markdown tables."""
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


def load(name):
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def table_fhrr(rows):
    Ms = [8, 16, 32, 64, 96, 128, 192]
    out = ["| D | " + " | ".join(f"M={m}" for m in Ms) + " |",
           "|---|" + "---|" * len(Ms)]
    for D in (64, 128, 256, 512, 1024):
        cells = []
        for m in Ms:
            hit = [r for r in rows if r["D"] == D and r["M"] == m]
            cells.append(f"{hit[0]['acc']:.2f}" if hit else "—")
        out.append(f"| {D} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_ghrr(rows):
    out = ["| config | coeffs | " + " | ".join(f"M={m}" for m in (8, 16, 32, 64, 96)) + " |",
           "|---|---|" + "---|" * 5]
    for D in (64, 128, 256):
        for m in (2, 3):
            for d in (0.0, 0.5):
                cells = []
                for M in (8, 16, 32, 64, 96):
                    hit = [r for r in rows if r["D"] == D and r["m"] == m
                           and r["diag"] == d and r["M"] == M]
                    cells.append(f"{hit[0]['acc']:.2f}" if hit else "—")
                out.append(f"| D={D}, m={m}, diag={d} | {D*m*m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_herr(rows):
    out = ["| decoder | k (non-zeros) | " + " | ".join(f"M={m}" for m in (8, 16, 32, 48, 64, 96)) + " |",
           "|---|---|" + "---|" * 6]
    for sysname in ("herr_oneshot", "herr_peel"):
        for k in (8, 16, 32, 64):
            cells = []
            for M in (8, 16, 32, 48, 64, 96):
                hit = [r for r in rows if r["system"] == sysname and r["extra_i"] == k and r["M"] == M]
                cells.append(f"{hit[0]['acc']:.2f}" if hit else "—")
            out.append(f"| {sysname} | {k} | " + " | ".join(cells) + " |")
    cells = []
    for M in (8, 16, 32, 48, 64, 96):
        hit = [r for r in rows if r["system"] == "herr_dense_peel" and r["M"] == M]
        cells.append(f"{hit[0]['acc']:.2f}" if hit else "—")
    out.append("| dense control | 128 | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    cap = load("capacity.jsonl")
    herr = load("herr.jsonl")
    noise = load("noise.jsonl")
    parts = []
    if cap:
        parts.append("### FHRR capacity (accuracy of role->filler retrieval)\n\n"
                     + table_fhrr([r for r in cap if r["system"] == "fhrr"]))
        parts.append("### GHRR capacity (left-unbind decoder)\n\n"
                     + table_ghrr([r for r in cap if r["system"] == "ghrr"]))
    if herr:
        parts.append("### HeRR capacity, D=128 fillers\n\n" + table_herr(herr))
    if noise:
        cells = [(r["extra_f"], r["acc"]) for r in noise]
        parts.append("### Noise robustness\n\n| SNR (dB) | " + " | ".join(f"{s:g}" for s, _ in cells)
                     + " |\n|---|" + "---|" * len(cells) + "\n| accuracy | "
                     + " | ".join(f"{a:.2f}" for _, a in cells) + " |")
    print("\n\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
