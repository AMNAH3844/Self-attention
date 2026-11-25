import math
import torch
import torch.nn as nn

import torch.nn as nn


class SelfAttention(nn.Module):
    """Implements a single-head self-attention layer."""

    def __init__(self, embed_dim: int):
        # Initialize the parent PyTorch nn.Module class
        super().__init__()

        # Store the dimensionality of the input and output embeddings
        self.embed_dim = embed_dim

        # Linear projection layers to transform input tokens into Queries, Keys, and Values
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Value projection

        # Final linear layer to project the weighted values back to the embedding space
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ v
        return self.out_proj(context)


# Example usage
x = torch.randn(2, 4, 8)
attn = SelfAttention(embed_dim=8)
out = attn(x)
print(out.shape)  # torch.Size([2, 4, 8])
