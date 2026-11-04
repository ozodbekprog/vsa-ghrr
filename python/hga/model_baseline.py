"""Plain transformer language model: the BPB control for the word-level model.

This is deliberately the most boring LM that can be trained on the same data and
compared in bits-per-byte against the GHRR word-level model: learned token
embedding + learned absolute positional embedding, pre-LN blocks with eager
scaled dot-product attention and an explicit causal mask, GELU FFN, final
LayerNorm, and an lm_head tied to the token embedding.  No GHRR binding, no
RoPE, no GQA, no MoE -- nothing that would confound the BPB comparison.

Parameter count at the default size (vocab 16000, d_model 384, 6 blocks,
FFN 1536, max_pos 1024, tied embeddings):

    16000*384 (tok) + 1024*384 (pos) + 6 * 1_774_464 (blocks) + 2*384 (ln_f)
    = 17_184_768
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["BaselineLM", "Block", "CausalSelfAttention"]


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention, eager, with an additive causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T, C) -> (B, H, T, Dh)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
        # Causal mask: position t may only attend to positions <= t.
        future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(diagonal=1)
        att = att.masked_fill(future, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                      # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + ffn(ln(x))."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.fc_in = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc_out = nn.Linear(ffn_dim, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        h = F.gelu(self.fc_in(self.ln_2(x)))
        x = x + self.dropout(self.fc_out(h))
        return x


class BaselineLM(nn.Module):
    """Decoder-only transformer LM with tied input/output embeddings."""

    def __init__(self, vocab_size: int, d_model: int = 384, n_blocks: int = 6,
                 n_heads: int = 6, ffn_dim: int = 1536, dropout: float = 0.0,
                 max_pos: int = 1024):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_blocks = int(n_blocks)
        self.n_heads = int(n_heads)
        self.max_pos = int(max_pos)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_pos, self.d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(self.d_model, self.n_heads, ffn_dim, dropout) for _ in range(self.n_blocks)
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual-output projections by 1/sqrt(2*L).
        resid_std = 0.02 / math.sqrt(2 * self.n_blocks)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

        # Tied embeddings: lm_head IS the transposed token embedding.
        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _check_length(self, T: int) -> None:
        if T > self.max_pos:
            raise ValueError(f"sequence length {T} exceeds max_pos={self.max_pos}")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] long -> logits [B, T, vocab_size]."""
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids of shape [B, T], got {tuple(token_ids.shape)}")
        B, T = token_ids.shape
        self._check_length(T)
        pos = torch.arange(T, device=token_ids.device)
        x = self.drop(self.tok_emb(token_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[N, d_model]: token + absolute position embeddings entering block 0.

        Accepts [T] or [B, T]; the batch/time axes are flattened so callers can
        index words without caring about the sequence layout.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids of shape [T] or [B, T], got {tuple(token_ids.shape)}")
        T = token_ids.shape[-1]
        self._check_length(T)
        pos = torch.arange(T, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        return x.reshape(-1, self.d_model)

    def output_embeddings(self) -> torch.Tensor:
        """[vocab, d_model]: lm_head rows (the same tensor as tok_emb.weight)."""
        return self.lm_head.weight
