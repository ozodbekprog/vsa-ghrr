"""HGA-1: FHRR word vectors -> learned projection -> GHRR transformer.

The input is *not* a learned token embedding table.  Every vocabulary word owns
a fixed FHRR character vector (2*512 real numbers: real/imag parts of 512
complex dimensions), produced by the C++ char encoder.  A small MLP projects
that 1024-d real vector into the transformer width, so the model never sees an
unconstrained per-word lookup at the input side.

Positions are bound into q/k (never v) by a single shared
`GHRRPositionBinder` with block_size=4, diagonality=1.0 and learnable=False.
With m=4 the binder folds two RoPE pairs into one 4x4 block; diagonality=1.0
makes every block a pure rotation expm(pos * G), i.e. a fixed relative-position
encoding that satisfies E(a)E(b) = E(a+b).  Learnable extra generators were
measured not to help at the experiment's token budgets, so they are off.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn

try:
    from gpuvsa.attention import GHRRPositionBinder
except ModuleNotFoundError:  # allow `python python/hga/model_hga.py` style use
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpuvsa.attention import GHRRPositionBinder

FHRR_DIM = 512          # complex dimensions per FHRR word vector
FHRR_REAL_DIM = 2 * FHRR_DIM


class WordProjection(nn.Module):
    """[B, W, 1024] real FHRR word vector -> [B, W, d_model]."""

    def __init__(self, d_fhrr: int = FHRR_DIM, d_model: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(2 * d_fhrr, 512)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(512, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2*D_fhrr]
        return self.fc2(self.act(self.fc1(x)))


class GHRRBlock(nn.Module):
    """Pre-LN transformer block with position-bound (GHRR) attention."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, ffn_dim: int = 1536,
                 block_size: int = 4, dropout: float = 0.0,
                 binder: GHRRPositionBinder | None = None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if binder is None:
            binder = GHRRPositionBinder(
                self.head_dim, block_size=block_size, diagonality=1.0,
                rope_theta=10000.0, learnable=False,
            )
        self.binder = binder

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _attend(self, q, k, v, E):
        B, L, H, Dh = q.shape
        q = self.binder.bind(q, E).transpose(1, 2)          # [B, H, L, Dh]
        k = self.binder.bind(k, E).transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal
        # fp32 softmax keeps short-sequence training stable in bf16/fp16 runs.
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = self.drop(probs)
        out = torch.matmul(probs, v)                          # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, H * Dh)
        return self.o_proj(out)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: [B, L] long, the token positions 0..L-1.
        hn = self.ln1(h)
        B, L, _ = hn.shape
        q = self.q_proj(hn).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(hn).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(hn).view(B, L, self.n_heads, self.head_dim)
        E = self.binder(position_ids)
        h = h + self._attend(q, k, v, E)
        h = h + self.drop(self.ffn(self.ln2(h)))
        return h


class HGA1Model(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_blocks=6, n_heads=6,
                 block_size=4, ffn_dim=1536, dropout=0.0, char_features=None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        if head_dim % block_size != 0:
            raise ValueError("head_dim must be divisible by block_size")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.word_proj = WordProjection(FHRR_DIM, d_model)

        # FHRR word vectors are data, not weights: a non-trainable buffer.
        # They move with .to(device) and are saved in the checkpoint so eval
        # does not silently depend on the data pipeline being at hand.
        if char_features is not None:
            if tuple(char_features.shape) != (vocab_size, FHRR_REAL_DIM):
                raise ValueError(
                    f"char_features must be [{vocab_size}, {FHRR_REAL_DIM}], "
                    f"got {tuple(char_features.shape)}"
                )
            self.register_buffer("char_features", char_features.float().clone())
            self.word_embedding = None
        else:
            # Standalone fallback (unit tests); the real pipeline passes
            # precomputed char_features.
            self.register_buffer("char_features", None)
            self.word_embedding = nn.Embedding(vocab_size, FHRR_REAL_DIM)

        # One binder (one role space) shared by every block, q/k only.
        self.binder = GHRRPositionBinder(
            head_dim, block_size=block_size, diagonality=1.0,
            rope_theta=10000.0, learnable=False,
        )
        self.blocks = nn.ModuleList([
            GHRRBlock(d_model, n_heads, ffn_dim, block_size, dropout, binder=self.binder)
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Output side is a plain projection; it is intentionally tied to nothing
        # (the input is a projection over fixed FHRR vectors, not an embedding).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def input_embeddings(self, word_ids: torch.Tensor) -> torch.Tensor:
        """word_ids: [B, W] or [N] long -> [N, d_model] projected vectors."""
        ids = word_ids.reshape(-1)
        if self.char_features is not None:
            x = self.char_features[ids]
        else:
            x = self.word_embedding(ids)
        return self.word_proj(x)

    def output_embeddings(self) -> torch.Tensor:
        """[vocab, d_model] rows of the output projection."""
        return self.lm_head.weight

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # word_ids: [B, W] long -> logits [B, W, vocab]
        B, W = word_ids.shape
        h = self.input_embeddings(word_ids).view(B, W, self.d_model)
        pos = torch.arange(W, device=word_ids.device).unsqueeze(0).expand(B, -1)
        for block in self.blocks:
            h = block(h, pos)
        return self.lm_head(self.ln_f(h))
