"""Noesis transformer — a from-scratch decoder-only LLM.

Modern recipe, zero borrowed weights:
  - RMSNorm (pre-norm, fp32 precision)
  - RoPE positional embeddings
  - Grouped-Query Attention (GQA) with causal masking + KV-cache
  - SwiGLU MLP (no biases anywhere)
  - Tied input/output embeddings
  - Residual-scaled init (std / sqrt(2*n_layers))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12          # query heads
    n_kv_heads: int = 8        # key/value heads (GQA; == n_heads -> classic MHA)
    head_dim: int = 64
    vocab_size: int = 32768
    max_seq_len: int = 1024
    mlp_hidden: int = 2816     # SwiGLU hidden size
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name: d[f.name] for f in __import__("dataclasses").fields(cls) if f.name in d}
        return cls(**known)

    def n_params(self, tied=True) -> int:
        emb = self.vocab_size * self.dim
        per_layer = 0
        per_layer += self.dim * (self.n_heads * self.head_dim)            # Wq
        per_layer += self.dim * (self.n_kv_heads * self.head_dim)         # Wk
        per_layer += self.dim * (self.n_kv_heads * self.head_dim)         # Wv
        per_layer += (self.n_heads * self.head_dim) * self.dim            # Wo
        per_layer += self.dim * self.mlp_hidden + self.dim * self.mlp_hidden + self.mlp_hidden * self.dim
        total = emb + per_layer * self.n_layers
        return total


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype, offset: int = 0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, hd]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c = cos[None, None, : x.shape[-2], :]
    s = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * c[..., : x1.shape[-1]] - x2 * s[..., : x1.shape[-1]],
                      x1 * s[..., x1.shape[-1]:] + x2 * c[..., x1.shape[-1]:]], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.q = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.hd)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)   # [B,H,T,hd]
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)      # [B,KV,T,hd]
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            pk, pv = cache                                                # [B,KV,Tprev,hd] or None
            if pk is not None and pk.numel():
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
            cache[0], cache[1] = k, v

        # GQA: repeat kv heads to match q heads
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        Tkv = k.shape[-2]
        y = torch.matmul(q, k.transpose(-2, -1)) * self.scale             # [B,H,T,Tkv]
        # causal mask in GLOBAL positions: query at pos (Tkv-T+t) attends j <= (Tkv-T+t)
        q_pos = torch.arange(Tkv - T, Tkv, device=x.device)
        k_pos = torch.arange(Tkv, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]                         # [T, Tkv]
        y = y.masked_fill(~causal, float("-inf"))
        y = F.softmax(y, dim=-1)
        y = torch.matmul(y, v)                                            # [B,H,T,hd]
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)  # gate
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.dim, bias=False)  # down
        self.w3 = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)  # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        x = x + self.mlp(self.n2(x))
        return x


class NoesisModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)

        # init: normal(0.02), residual projections scaled
        self.apply(self._init)
        for name, p in self.named_parameters():
            if any(k in name for k in ("q.weight", "k.weight", "v.weight", "o.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, caches=None):
        B, T = idx.shape
        dev = idx.device
        past = caches[0][0].shape[-2] if (caches is not None and caches[0] is not None and caches[0][0] is not None and caches[0][0].numel()) else 0
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, dev, idx.dtype, offset=past)

        x = self.embed(idx)
        if caches is None:
            caches = [None] * self.cfg.n_layers
        for i, blk in enumerate(self.blocks):
            if caches[i] is None:
                caches[i] = [None, None]
            x = blk(x, cos, sin, caches[i])
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight)             # tied embeddings
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1).long())
        return logits, loss

    def init_caches(self, batch_size: int, device):
        return [[None, None] for _ in range(self.cfg.n_layers)]
