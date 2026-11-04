"""Model retrofit: swap attention for GHRR attention and measure perplexity."""
from __future__ import annotations

import json
import math
import time

import torch


def load_model(path_or_name: str, dtype=torch.float16, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForCausalLM.from_pretrained(path_or_name, torch_dtype=dtype)
    model.config.use_cache = False  # GHRR attention has no KV cache (yet)
    model.to(device)
    model.eval()
    return model, tok


def encode_corpus(tok, texts, eos_id=None):
    ids: list[int] = []
    eos_id = tok.eos_token_id if eos_id is None else eos_id
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if eos_id is not None:
            ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, max_len: int = 1024, stride: int = 512,
               max_tokens: int | None = None, device: str = "cuda", log=print):
    """Sliding-window perplexity, counting each token exactly once."""
    model.eval()
    for m in model.modules():
        if hasattr(m, "clear_cache"):
            m.clear_cache()
    n = ids.numel()
    if max_tokens:
        n = min(n, max_tokens)
    ids = ids[:n].to(device)
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    begin = 0
    while begin < n - 1:
        end = min(begin + max_len, n)
        chunk = ids[begin:end].unsqueeze(0)
        target = chunk.clone()
        if begin > 0:
            target[:, : max(0, chunk.size(1) - stride)] = -100
        out = model(input_ids=chunk, labels=target)
        counted = int((target != -100).sum())
        total_nll += float(out.loss) * counted
        total_tok += counted
        if log and (begin // stride) % 20 == 0:
            log(f"    ppl window {begin}/{n}  elapsed {time.time()-t0:.0f}s")
        begin += stride
    return math.exp(total_nll / max(1, total_tok)), total_tok, time.time() - t0


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class LoRALinear(torch.nn.Module):
    """Minimal LoRA wrapper: y = W x + (alpha/r) * B A x, with A, B in fp32."""

    def __init__(self, base, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        import math

        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        delta = (self.drop(x.float()) @ self.A.T) @ self.B.T
        return out + (delta * self.scaling).to(out.dtype)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
               targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Wrap every matching Linear inside the decoder layers. Returns the count."""
    n = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in targets:
            mod = getattr(attn, name, None)
            if isinstance(mod, torch.nn.Linear):
                setattr(attn, name, LoRALinear(mod, r, alpha, dropout))
                n += 1
    return n


def finetune_lora(model, ids: torch.Tensor, binder, steps: int = 200, seq_len: int = 512,
                  lr_lora: float = 2e-4, lr_gen: float = 1e-2, device: str = "cuda",
                  log=print, seed: int = 0, r: int = 16, alpha: int = 32):
    """LoRA on the attention projections; the GHRR generators are trained directly."""
    n_wrapped = apply_lora(model, r=r, alpha=alpha)
    log(f"    LoRA wrapped {n_wrapped} projections (r={r}, alpha={alpha})")
    model.train()
    for m in model.modules():
        if hasattr(m, "clear_cache"):
            m.clear_cache()

    gen_params = []
    if binder is not None:
        for p in binder.parameters():
            p.requires_grad_(True)
            gen_params.append(p)
    lora_params = [p for n, p in model.named_parameters()
                   if (n.endswith(".A") or n.endswith(".B")) and p.requires_grad]

    groups = [{"params": lora_params, "lr": lr_lora, "weight_decay": 0.0}]
    if gen_params:
        groups.append({"params": gen_params, "lr": lr_gen, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    g = torch.Generator().manual_seed(seed)
    n = ids.numel()
    t0 = time.time()
    losses = []
    for step in range(steps):
        start = int(torch.randint(0, max(1, n - seq_len - 1), (1,), generator=g).item())
        chunk = ids[start: start + seq_len].unsqueeze(0).to(device)
        out = model(input_ids=chunk, labels=chunk)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for grp in groups for p in grp["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 20 == 0:
            log(f"    step {step}/{steps} loss {float(loss):.4f} "
                f"({(time.time()-t0)/(step+1):.2f}s/step)")
    return model, losses


def save_report(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
