"""Correctness tests for GHRR attention (CPU, no downloads).

Run:  python tests/test_attention.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from gpuvsa.attention import GHRRPositionBinder, install_ghrr  # noqa: E402

FAIL = 0


def check(name, ok, value=0.0):
    global FAIL
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<48} {value}")
    if not ok:
        FAIL += 1


def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m, cfg


def main():
    m, cfg = tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        base = m(ids).logits.clone()

    print("[equivalence]")
    binder = install_ghrr(m, block_size=2, diagonality=1.0, verbose=False)
    with torch.no_grad():
        out = m(ids).logits
    d = (out - base).abs().max().item()
    check("diagonality=1.0 reproduces the original model", d < 1e-4, d)

    m2, _ = tiny_model()
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if "self_attn" not in k},
                       strict=False)
    binder2 = install_ghrr(m2, block_size=2, diagonality=0.5, verbose=False)
    with torch.no_grad():
        out2 = m2(ids).logits
    d2 = (out2 - base).abs().max().item()
    check("diagonality=0.5 also starts exactly at RoPE", d2 < 1e-4, d2)

    print("[algebra]")
    check("binder has trainable generators", sum(p.numel() for p in binder2.parameters()) > 0,
          sum(p.numel() for p in binder2.parameters()))
    # The extra generators must be zero at init so the warm start is exact.
    check("extra generators initialise to zero",
          binder2.extra is not None and binder2.extra.abs().max().item() == 0.0)

    # E(pos) must be orthogonal and satisfy the group law E(a)E(b)=E(a+b).
    pos = torch.arange(0, 8).unsqueeze(0)
    E = binder2(pos)                                  # (1,8,D,m,m)
    err = (E @ E.transpose(-1, -2) - torch.eye(2).expand_as(E @ E.transpose(-1, -2))).abs().max()
    check("E(pos) is orthogonal", err.item() < 1e-5, err.item())
    E7 = binder2(torch.tensor([[7]]))
    E3 = binder2(torch.tensor([[3]]))
    E4 = binder2(torch.tensor([[4]]))
    g = (E3 @ E4 - E7).abs().max().item()
    check("group law E(3)E(4)=E(7)", g < 1e-5, g)

    print("[causality]")
    m.eval()
    with torch.no_grad():
        a = m(ids).logits
        ids2 = ids.clone()
        ids2[:, -1] = (ids2[:, -1] + 7) % 256
        b = m(ids2).logits
    same_prefix = (a[:, :-1] - b[:, :-1]).abs().max().item()
    changed_last = (a[:, -1] - b[:, -1]).abs().max().item()
    check("future token does not leak backwards", same_prefix < 1e-4, same_prefix)
    check("last position does change", changed_last > 1e-5, changed_last)

    print("[gradients]")
    m2.train()
    binder2.zero_grad(set_to_none=True)
    out = m2(ids, labels=ids)
    out.loss.backward()
    # The extra generators start at zero, but their gradient must be non-zero.
    if binder2.extra is not None:
        grad = binder2.extra.grad
        check("generator receives gradient", grad is not None and grad.abs().max() > 0,
              float(grad.abs().max()) if grad is not None else -1.0)
        # RoPE part must stay fixed (it is a buffer, not a parameter)
        check("rope generators are frozen buffers", "G_rope" in dict(binder2.named_buffers()))

    print("[positions]")
    m.eval()
    with torch.no_grad():
        p1 = m(ids, position_ids=torch.arange(24).unsqueeze(0).expand(2, -1)).logits
        p2 = m(ids, position_ids=torch.arange(24).unsqueeze(0).expand(2, -1)).logits
    check("position encoding is deterministic", (p1 - p2).abs().max().item() == 0.0)

    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURES'}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
