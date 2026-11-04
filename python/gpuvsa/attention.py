"""GHRR-structured self-attention.

The GHRR algebra (see the C++ core) binds a vector with a role by multiplying
m x m matrices elementwise.  Here the "role" is the token position, and the
binding is a real orthogonal block matrix acting on blocks of `block_size`
consecutive head dimensions:

    q'_b = E_b(pos) q_b,     E_b(pos) = expm(pos * G_b)

with G_b real skew-symmetric.  Because E is a one-parameter subgroup
(E(a) E(b) = E(a+b)), the score <q'_i, k'_j> depends only on (pos_j - pos_i),
exactly like RoPE -- but with m-dimensional blocks instead of 2-D rotations.

`diagonality` controls how much of the binding is the pure rotation (RoPE)
part versus a learnable non-commutative mixing term:

    diagonality = 1.0  -> every block is a pure rotation (bit-exact RoPE)
    diagonality = 0.5  -> half of the blocks also carry a learnable
                          skew-symmetric generator, initialised at zero, so
                          the model still starts exactly at RoPE but can
                          learn block-internal mixing.

That warm start is what makes the retrofit verifiable: with diagonality=1.0
the retrofitted model must reproduce the original perplexity to within
floating-point noise.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _rope_permutation(head_dim: int, device) -> torch.Tensor:
    """Llama/Qwen RoPE pairs dim i with dim i + head_dim/2, not (2i, 2i+1).

    We reorder the head vector so that each rotated pair is contiguous.
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long, device=device)
    perm[0::2] = torch.arange(0, half, device=device)
    perm[1::2] = torch.arange(half, head_dim, device=device)
    return perm


class GHRRPositionBinder(nn.Module):
    """Shared positional binding for every layer (one role space per model)."""

    def __init__(self, head_dim: int, block_size: int = 2, diagonality: float = 1.0,
                 rope_theta: float = 10000.0, learnable: bool = True):
        super().__init__()
        if head_dim % block_size != 0:
            raise ValueError("head_dim must be divisible by block_size")
        if block_size % 2 != 0:
            raise ValueError("block_size must be even (RoPE blocks are 2-D rotations)")
        self.head_dim = head_dim
        self.block_size = block_size
        self.diagonality = float(diagonality)
        self.num_blocks = head_dim // block_size
        m = block_size

        # Fixed RoPE generators, folded into blocks of size m.
        G = torch.zeros(self.num_blocks, m, m, dtype=torch.float32)
        J = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        for b in range(self.num_blocks):
            for half in range(m // 2):
                pair = b * (m // 2) + half          # index into inv_freq
                theta = rope_theta ** (-2.0 * pair / head_dim)
                G[b, 2 * half:2 * half + 2, 2 * half:2 * half + 2] = theta * J
        self.register_buffer("G_rope", G, persistent=False)

        # Learnable non-commutative part on a fraction of the blocks.
        keep = torch.zeros(self.num_blocks, 1, 1)
        n_extra = int(round((1.0 - self.diagonality) * self.num_blocks))
        if n_extra > 0:
            keep[:n_extra] = 1.0
        self.register_buffer("_extra_mask", keep, persistent=False)
        extra = torch.zeros(self.num_blocks, m, m)
        self.extra = nn.Parameter(extra) if learnable else None

        self.register_buffer("_perm", _rope_permutation(head_dim, torch.device("cpu")),
                             persistent=False)
        self._cache: dict[tuple, torch.Tensor] = {}

    def _skew(self) -> torch.Tensor:
        if self.extra is None:
            return self.G_rope
        e = self.extra * self._extra_mask
        e = 0.5 * (e - e.transpose(-1, -2))       # project to skew-symmetric
        return self.G_rope + e

    def clear_cache(self):
        """Must be called after any parameter update (eval caches E)."""
        self._cache = {}

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """position_ids: (B, L) long -> E: (B, L, D, m, m) orthogonal."""
        B, L = position_ids.shape
        # With no learnable generators the encoding is constant, so caching is
        # always safe; otherwise only cache in eval (params may change in train).
        use_cache = (self.extra is None) or (not self.training)
        key = (B, L, str(position_ids.device), position_ids.dtype,
               int(position_ids[0, 0].item()), int(position_ids[0, -1].item()))
        if use_cache and key in self._cache:
            return self._cache[key]

        G = self._skew()
        pos = position_ids.to(torch.float32)
        # angles: (B, L, D)
        # pos (B,L) -> (B,L,1,1,1);  G (D,m,m) -> (1,1,D,m,m)
        ang = pos.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * G.unsqueeze(0).unsqueeze(0)
        E = torch.linalg.matrix_exp(ang)            # (B, L, D, m, m)

        if use_cache:
            self._cache = {key: E}
        return E

    def bind(self, x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """x: (B, L, H, head_dim) -> bound (B, L, H, D, m)."""
        B, L, H, _ = x.shape
        m = self.block_size
        x = x[..., self._perm]
        x = x.view(B, L, H, self.num_blocks, m)
        out = torch.einsum("bldij,blhdj->blhdi", E.to(x.dtype), x)
        return out.reshape(B, L, H, self.head_dim)


class GHRRAttention(nn.Module):
    """Drop-in replacement for Llama/Qwen2 style self-attention."""

    def __init__(self, config, layer_idx: int = 0, binder: GHRRPositionBinder | None = None,
                 qkv_bias: bool = False, o_bias: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=o_bias)

        self.binder = binder if binder is not None else GHRRPositionBinder(
            self.head_dim,
            block_size=2,
            diagonality=1.0,
            rope_theta=getattr(config, "rope_theta", 10000.0),
        )

    @classmethod
    def from_attention(cls, attn: nn.Module, config, layer_idx: int, binder):
        # Qwen2 has bias in q/k/v (attention_bias=True); Llama does not.
        qkv_bias = getattr(attn.q_proj, "bias", None) is not None
        o_bias = getattr(attn.o_proj, "bias", None) is not None
        new = cls(config, layer_idx=layer_idx, binder=binder, qkv_bias=qkv_bias, o_bias=o_bias)
        with torch.no_grad():
            new.q_proj.weight.copy_(attn.q_proj.weight)
            new.k_proj.weight.copy_(attn.k_proj.weight)
            new.v_proj.weight.copy_(attn.v_proj.weight)
            new.o_proj.weight.copy_(attn.o_proj.weight)
            if qkv_bias:
                new.q_proj.bias.copy_(attn.q_proj.bias)
                new.k_proj.bias.copy_(attn.k_proj.bias)
                new.v_proj.bias.copy_(attn.v_proj.bias)
            if o_bias:
                new.o_proj.bias.copy_(attn.o_proj.bias)
        return new.to(attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        B, L, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        E = self.binder(position_ids)                     # (B, L, D, m, m)
        q = self.binder.bind(q, E)
        k = self.binder.bind(k, E)

        q = q.transpose(1, 2)                             # (B, H, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal
        if attention_mask is not None:
            mask = attention_mask
            if mask.dtype == torch.bool:
                mask = torch.where(mask, float("-inf"), 0.0)
            if mask.dtype != scores.dtype:
                mask = mask.to(scores.dtype)
            if mask.dim() == 4:
                scores = scores + mask[:, :, :L, :L]
            elif mask.dim() == 3:
                scores = scores + mask[:, :, :L].unsqueeze(-2)
            else:
                scores = scores + mask[:, None, None, :L]

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)                      # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        attn_weights = probs if output_attentions else None
        return out, attn_weights, past_key_value


def install_ghrr(model, block_size: int = 2, diagonality: float = 1.0,
                 rope_theta: float | None = None, verbose: bool = True):
    """Replace every decoder-layer self-attention with GHRRAttention.

    Returns the shared binder so the caller can optimise its parameters.
    """
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    binder = GHRRPositionBinder(
        hd,
        block_size=block_size,
        diagonality=diagonality,
        rope_theta=rope_theta or getattr(cfg, "rope_theta", 10000.0),
    )
    layers = model.model.layers
    for i, layer in enumerate(layers):
        old = layer.self_attn
        layer.self_attn = GHRRAttention.from_attention(old, cfg, i, binder)
        del old
    binder = binder.to(layers[0].self_attn.q_proj.weight.device,
                       dtype=layers[0].self_attn.q_proj.weight.dtype)
    for layer in layers:
        layer.self_attn.binder = binder
    if verbose:
        n = sum(p.numel() for p in binder.parameters())
        print(f"[ghrr] replaced {len(layers)} attention layers | block_size={block_size} "
              f"diagonality={diagonality} | binder params={n}")
    return binder
