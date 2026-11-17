"""GPT model components."""

import torch
from torch import nn


class Head(nn.Module):
    """Single-head causal self-attention."""

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


if __name__ == "__main__":
    torch.manual_seed(0)
    head = Head(dim=32, head_dim=8)
    x = torch.randn(2, 5, 32)
    out = head(x)
    print(f"x: {tuple(x.shape)} -> out: {tuple(out.shape)}")
