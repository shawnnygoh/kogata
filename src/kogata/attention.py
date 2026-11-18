"""Attention mixers.

Shapes use B for batch size and T for sequence length.
"""

import torch
from torch import nn


class Head(nn.Module):
    """Single-head causal self-attention.

    Each position attends to itself and to earlier positions only.

    Args:
        dim: Width of the input vectors.
        head_dim: Width of the query, key and value vectors, and of the output.

    Shape:
        - Input: (B, T, dim)
        - Output: (B, T, head_dim)
    """

    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, head_dim, bias=False)
        self.k_proj = nn.Linear(dim, head_dim, bias=False)
        self.v_proj = nn.Linear(dim, head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)  # (B, T, head_dim)
        v = self.v_proj(x)  # (B, T, head_dim)

        scores = q @ k.transpose(-2, -1)  # (B, T, T)
        scores = scores * self.head_dim**-0.5
        # True where token i may attend to token j, i.e. j <= i
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = scores.softmax(dim=-1)  # each row sums to 1
        out = weights @ v  # (B, T, head_dim)

        return out


class MHAWrapper(nn.Module):
    """Multi-head causal self-attention (MHA) as a list of single heads.

    Runs each `Head` separately, joins their outputs and projects them back
    to `dim` with `o_proj`.

    Args:
        dim: Width of the input and output vectors.
        n_heads: Number of heads.
        head_dim: Width of each head. Defaults to `dim // n_heads`.

    Raises:
        ValueError: If `head_dim` is not given and `n_heads` does not divide `dim`.

    Shape:
        - Input: (B, T, dim)
        - Output: (B, T, dim)
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int | None = None):
        super().__init__()
        self.dim = dim
        if head_dim is None:
            if dim % n_heads != 0:
                raise ValueError(f"dim={dim} is not divisible by n_heads={n_heads}")
            head_dim = dim // n_heads
        self.heads = nn.ModuleList(Head(dim, head_dim) for _ in range(n_heads))
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head gives (B, T, head_dim); joined: (B, T, n_heads * head_dim)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.o_proj(out)  # (B, T, dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = Head(dim=32, head_dim=8)
    x = torch.randn(2, 5, 32)
    print(f"Head:      x: {tuple(x.shape)} -> out: {tuple(head(x).shape)}")
    attn = MHAWrapper(dim=32, n_heads=4)
    print(f"Wrapper:   x: {tuple(x.shape)} -> out: {tuple(attn(x).shape)}")
