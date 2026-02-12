import torch
import torch.nn as nn
from inference.kv_cache.naive_cache import Cache
from inference.layers.attention.sdpa_attention import scaled_dot_product_attention


class Attention(nn.Module):
    """Attention with kv cache per attention layer"""

    def __init__(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_idx: int,
        scale: float = 1.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx
        self.scale = scale

        self.cache = Cache(
            tag=layer_idx,
            batch=batch,
            head_dim=head_dim,
            num_heads=num_kv_heads,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q= [B, Lq, nq, dh]
        KV = [B, Lkv, nkv, dh]
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        K, V = self.cache.update(k, v)

        out = scaled_dot_product_attention(
            query=q,
            key=K,
            value=V,
            is_causal=True,
            scale=self.scale,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )

        return out.transpose(1, 2)
