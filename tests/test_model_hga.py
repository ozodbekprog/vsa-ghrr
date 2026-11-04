"""Correctness tests for the HGA-1 model (CPU, no downloads).

Run:  python tests/test_model_hga.py
Exit code 0 = all pass, 1 = at least one failure.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hga.model_hga import HGA1Model  # noqa: E402

FAIL = 0


def check(name, ok, value=0.0):
    global FAIL
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<50} {value}")
    if not ok:
        FAIL += 1


def main():
    torch.manual_seed(0)
    V = 16000
    char = torch.randn(V, 1024) * 0.05
    model = HGA1Model(V, char_features=char)
    model.eval()

    print("[params]")
    n = sum(p.numel() for p in model.parameters())
    check("total params in 17-20M", 17_000_000 <= n <= 20_000_000, n)
    buffers = dict(model.named_buffers())
    check("char_features is a non-trainable buffer",
          "char_features" in buffers and buffers["char_features"].requires_grad is False,
          tuple(buffers["char_features"].shape))

    ids = torch.randint(0, V, (2, 17))

    print("[forward]")
    with torch.no_grad():
        logits = model(ids)
    check("logits [2,17,16000] -> [2,17,16000]",
          tuple(logits.shape) == (2, 17, V), tuple(logits.shape))

    print("[causality]")
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 12345) % V
    with torch.no_grad():
        logits2 = model(ids2)
    prefix = (logits[:, :-1] - logits2[:, :-1]).abs().max().item()
    last = (logits[:, -1] - logits2[:, -1]).abs().max().item()
    check("changing last word leaves earlier logits unchanged", prefix < 1e-6, prefix)
    check("last-position logits do change", last > 1e-6, last)

    print("[binder]")
    pos = torch.arange(17).unsqueeze(0)
    E1 = model.binder(pos)
    E2 = model.binder(pos)
    check("binder output shape (B,L,D,m,m)", tuple(E1.shape) == (1, 17, 16, 4, 4),
          tuple(E1.shape))
    check("binder identical across calls", torch.equal(E1, E2),
          (E1 - E2).abs().max().item())

    print("[embeddings]")
    ie = model.input_embeddings(torch.randint(0, V, (3, 7)))
    check("input_embeddings [N,d_model]", tuple(ie.shape) == (21, 384), tuple(ie.shape))
    oe = model.output_embeddings()
    check("output_embeddings [vocab,d_model]", tuple(oe.shape) == (V, 384), tuple(oe.shape))

    print("[gradients]")
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(ids)
    loss = F.cross_entropy(out.view(-1, V), ids.view(-1))
    loss.backward()
    g_proj = model.word_proj.fc1.weight.grad
    g_q0 = model.blocks[0].q_proj.weight.grad
    g_ffn_last = model.blocks[-1].ffn[-1].weight.grad
    g_head = model.lm_head.weight.grad
    check("grad reaches word projection", g_proj is not None and g_proj.abs().max() > 0,
          float(g_proj.abs().max()) if g_proj is not None else -1.0)
    check("grad reaches block 0 q_proj", g_q0 is not None and g_q0.abs().max() > 0,
          float(g_q0.abs().max()) if g_q0 is not None else -1.0)
    check("grad reaches last block FFN", g_ffn_last is not None and g_ffn_last.abs().max() > 0,
          float(g_ffn_last.abs().max()) if g_ffn_last is not None else -1.0)
    check("grad reaches lm_head", g_head is not None and g_head.abs().max() > 0,
          float(g_head.abs().max()) if g_head is not None else -1.0)
    check("no gradient on FHRR buffer",
          model.char_features.grad is None)

    print("[overfit 3 steps]")
    torch.manual_seed(1)
    small = HGA1Model(512, n_blocks=2, ffn_dim=768,
                      char_features=torch.randn(512, 1024) * 0.05)
    opt = torch.optim.Adam(small.parameters(), lr=1e-3)
    xs = torch.randint(0, 512, (4, 32))
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits = small(xs)
        loss = F.cross_entropy(logits.view(-1, 512), xs.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  step loss {losses[-1]:.4f}")
    check("3-step loss drops below initial", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    print("[fallback embedding]")
    m2 = HGA1Model(128, n_blocks=1, ffn_dim=256)
    out2 = m2(torch.randint(0, 128, (2, 8)))
    check("char_features=None path works", tuple(out2.shape) == (2, 8, 128),
          tuple(out2.shape))

    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURES'}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
