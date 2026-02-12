import torch


# naive kv-cache
class Cache:
    def __init__(
        self,
        tag: any,
        batch: int,
        head_dim: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        self.tag = tag

        seq_len = 0  # starting seq len is 0
        self.k_cache = torch.empty(
            (batch, num_heads, seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty(
            (batch, num_heads, seq_len, head_dim),
            device=device,
            dtype=dtype,
        )

    def update(self, k: torch.Tensor, v: torch.Tensor):

        self.k_cache = torch.cat([self.k_cache, k], dim=-2)
        self.v_cache = torch.cat([self.v_cache, v], dim=-2)

        return self.k_cache, self.v_cache
