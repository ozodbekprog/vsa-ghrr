"""Correctness tests for the plain-transformer BPB baseline (CPU/GPU, no downloads).

Checks: parameter count at the contract size, forward shapes, tied embeddings,
causal masking, and that gradients reach every parameter.

Run:  python tests/test_baseline.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import torch  # noqa: E402

try:  # `hga` is a namespace package; fall back to a direct module import
    from hga.model_baseline import BaselineLM  # noqa: E402
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(HERE, "..", "python", "hga"))
    from model_baseline import BaselineLM  # noqa: E402

VOCAB = 16000
D_MODEL = 384
N_BLOCKS = 6
N_HEADS = 6
FFN = 1536
MAX_POS = 1024

FAIL = 0


def check(name, ok, value=0.0):
    global FAIL
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<52} {value}")
    if not ok:
        FAIL += 1


def main():
    torch.manual_seed(0)
    m = BaselineLM(VOCAB, d_model=D_MODEL, n_blocks=N_BLOCKS, n_heads=N_HEADS,
                   ffn_dim=FFN, dropout=0.0, max_pos=MAX_POS)
    n_params = sum(p.numel() for p in m.parameters())

    print("[params]")
    expected = (VOCAB * D_MODEL + MAX_POS * D_MODEL + 2 * D_MODEL
                + N_BLOCKS * (4 * (D_MODEL * D_MODEL + D_MODEL)
                              + 4 * D_MODEL  # 2 LayerNorms: weight + bias each
                              + (D_MODEL * FFN + FFN)
                              + (FFN * D_MODEL + D_MODEL)))
    check("param count matches analytic value", n_params == expected, f"{n_params} vs {expected}")
    check("params in 17-20M", 17_000_000 <= n_params <= 20_000_000, n_params)

    print("[tied embeddings]")
    tied_same_obj = m.lm_head.weight is m.tok_emb.weight
    check("lm_head.weight is tok_emb.weight", tied_same_obj)
    check("tied tensors share data_ptr",
          m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr())
    check("tied tensors allclose",
          torch.allclose(m.lm_head.weight, m.tok_emb.weight))
    check("output_embeddings shape", tuple(m.output_embeddings().shape) == (VOCAB, D_MODEL),
          tuple(m.output_embeddings().shape))
    check("output_embeddings is the tied weight",
          m.output_embeddings().data_ptr() == m.tok_emb.weight.data_ptr())

    print("[shapes]")
    m.eval()
    B, T = 2, 16
    ids = torch.randint(4, VOCAB, (B, T))
    with torch.no_grad():
        logits = m(ids)
    check("forward -> [B,T,vocab]",
          tuple(logits.shape) == (B, T, VOCAB) and logits.dtype == torch.float32,
          tuple(logits.shape))
    emb = m.input_embeddings(ids)
    check("input_embeddings [N,d_model]", tuple(emb.shape) == (B * T, D_MODEL), tuple(emb.shape))
    emb1d = m.input_embeddings(torch.tensor([5, 6, 7]))
    check("input_embeddings accepts 1-D ids", tuple(emb1d.shape) == (3, D_MODEL),
          tuple(emb1d.shape))
    check("logits are finite", bool(torch.isfinite(logits).all()))

    print("[causality]")
    with torch.no_grad():
        a = m(ids)
        ids2 = ids.clone()
        ids2[:, -1] = (ids2[:, -1] + 7) % VOCAB
        b = m(ids2)
    prefix = (a[:, :-1] - b[:, :-1]).abs().max().item()
    last = (a[:, -1] - b[:, -1]).abs().max().item()
    check("changing last token leaves earlier logits unchanged", prefix < 1e-6, prefix)
    check("last position does change", last > 1e-5, last)
    # Position t must not depend on anything after t, not just the final token.
    with torch.no_grad():
        ids3 = ids.clone()
        ids3[:, T // 2:] = (ids3[:, T // 2:] + 11) % VOCAB
        c = m(ids3)
    mid = (a[:, :T // 2] - c[:, :T // 2]).abs().max().item()
    check("suffix change leaves prefix logits unchanged", mid < 1e-6, mid)

    print("[max_pos guard]")
    try:
        with torch.no_grad():
            m(torch.zeros(1, MAX_POS + 1, dtype=torch.long))
        raised = False
    except ValueError:
        raised = True
    check("sequence longer than max_pos raises", raised)

    print("[gradients]")
    m.train()
    m.zero_grad(set_to_none=True)
    logits = m(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), ids.reshape(-1))
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    zero = [n for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().max().item() == 0.0]
    check("loss is finite", bool(torch.isfinite(loss)), float(loss.detach()))
    check("every parameter receives a gradient", not missing, missing[:3])
    check("no all-zero gradients", not zero, zero[:3])
    check("token embedding gradient is non-zero",
          m.tok_emb.weight.grad is not None and m.tok_emb.weight.grad.abs().max().item() > 0,
          float(m.tok_emb.weight.grad.abs().max()) if m.tok_emb.weight.grad is not None else -1.0)
    check("position embedding gradient is non-zero",
          m.pos_emb.weight.grad is not None and m.pos_emb.weight.grad.abs().max().item() > 0,
          float(m.pos_emb.weight.grad.abs().max()) if m.pos_emb.weight.grad is not None else -1.0)
    check("tied weights accumulate one shared gradient",
          m.lm_head.weight.grad is m.tok_emb.weight.grad)

    print(f"\nparam count: {n_params:,}")
    print(f"{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURES'}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
