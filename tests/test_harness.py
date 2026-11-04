#!/usr/bin/env python
"""Fast CPU smoke test for python/hga/{train,eval}.py.

Builds a 64-word synthetic corpus (4k train units), trains the tiny HGA-1
(HGA_TEST_TINY=1 -> d_model=64, 2 blocks, 2 heads) for 5 steps, checks
resume, then evaluates and asserts ppl is finite and bpb > 0.  The baseline
path is smoke-tested the same way.  Skips cleanly if the model files that
other agents own are not present yet.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "python" / "hga" / "train.py"
EVAL = ROOT / "python" / "hga" / "eval.py"
MODELS = [ROOT / "python" / "hga" / "model_hga.py",
          ROOT / "python" / "hga" / "model_baseline.py"]
VOCAB = 64
TRAIN_UNITS = 4000
TEST_UNITS = 1000
SEMANTIC = ["olma", "meva", "stol", "daraxt", "non", "suv", "kitob", "maktab"]


def skip(msg: str) -> None:
    print(f"SKIP: {msg}")
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)


def make_data(root: Path) -> Path:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    words = {}
    for split, n in (("train", TRAIN_UNITS), ("test", TEST_UNITS)):
        ids = rng.integers(12, VOCAB, size=n, dtype=np.int32)
        ids[::37] = 3  # sprinkle <eos>, as the real data does
        words[split] = ids
        np.save(data / f"words_{split}.npy", ids)
        np.save(data / f"bpe_{split}.npy",
                (ids % 64).astype(np.uint16))

    itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + SEMANTIC + \
        [f"w{i}" for i in range(4 + len(SEMANTIC), VOCAB)]
    (data / "word_vocab.json").write_text(json.dumps(
        {"itos": itos, "stoi": {w: i for i, w in enumerate(itos)}}))

    (data / "meta.json").write_text(json.dumps({
        "bytes": {"train": TRAIN_UNITS * 5, "test": TEST_UNITS * 5},
        "units": {"train": TRAIN_UNITS, "test": TEST_UNITS},
    }))
    return data


def run(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ, HGA_TEST_TINY="1")
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise AssertionError(f"command failed: {' '.join(cmd)}")
    return proc


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_eval(path: Path, expect_semantic: bool) -> dict:
    result = json.loads(path.read_text())
    assert np.isfinite(result["ppl"]) and result["ppl"] > 0, result
    assert result["bpb"] is not None and result["bpb"] > 0, result
    assert result["n_units"] > 0 and result["ce_sum"] > 0, result
    status = result["semantic"]["status"]
    if expect_semantic:
        assert status == "ok", result["semantic"]
    else:
        assert status == "skipped", result["semantic"]
    return result


def main() -> None:
    missing = [str(p) for p in MODELS if not p.exists()]
    if missing:
        skip(f"model files not present yet: {missing}")
        return

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hga_harness_") as tmp:
        tmp = Path(tmp)
        data = make_data(tmp)
        hga_ckpt = tmp / "hga.pt"
        hga_log = tmp / "hga.jsonl"

        common = ["--data", str(data), "--device", "cpu", "--dtype", "fp32",
                  "--batch", "8", "--seq", "64", "--lr", "1e-3", "--warmup", "2",
                  "--log-every", "1", "--save-every", "2", "--seed", "0"]

        # 1) HGA: 3 steps, then resume to 5 -> exercises exact-resume path.
        run([str(TRAIN), "--model", "hga", "--out", str(hga_log),
             "--ckpt", str(hga_ckpt), "--steps", "3", *common])
        assert hga_ckpt.exists(), "checkpoint not written"
        run([str(TRAIN), "--model", "hga", "--out", str(hga_log),
             "--ckpt", str(hga_ckpt), "--steps", "5", "--resume", *common])

        import torch  # imported lazily so the skip path stays fast
        state = torch.load(hga_ckpt, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "step", "model_args"):
            assert key in state, f"checkpoint missing {key}"
        assert state["step"] == 5, state["step"]

        logs = read_jsonl(hga_log)
        assert {rec["step"] for rec in logs} >= {1, 2, 3, 4, 5}, logs
        for rec in logs:
            for key in ("step", "loss", "lr", "tokens_per_s", "wall_s"):
                assert key in rec, f"log line missing {key}: {rec}"

        eval_hga = tmp / "eval_hga.json"
        run([str(EVAL), "--model", "hga", "--ckpt", str(hga_ckpt),
             "--data", str(data), "--split", "test", "--device", "cpu",
             "--batch", "8", "--max-units", "512", "--out", str(eval_hga)])
        res_hga = check_eval(eval_hga, expect_semantic=True)

        # 2) Baseline path.
        base_ckpt = tmp / "baseline.pt"
        base_log = tmp / "baseline.jsonl"
        run([str(TRAIN), "--model", "baseline", "--out", str(base_log),
             "--ckpt", str(base_ckpt), "--steps", "3", *common])
        eval_base = tmp / "eval_base.json"
        run([str(EVAL), "--model", "baseline", "--ckpt", str(base_ckpt),
             "--data", str(data), "--split", "test", "--device", "cpu",
             "--batch", "8", "--max-units", "512", "--out", str(eval_base)])
        res_base = check_eval(eval_base, expect_semantic=False)

        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, f"smoke test too slow: {elapsed:.1f}s"
        print(f"OK  hga: ppl={res_hga['ppl']:.2f} bpb={res_hga['bpb']:.4f} "
              f"n_units={res_hga['n_units']} | baseline: "
              f"ppl={res_base['ppl']:.2f} bpb={res_base['bpb']:.4f} "
              f"| {elapsed:.1f}s")


if __name__ == "__main__":
    main()
