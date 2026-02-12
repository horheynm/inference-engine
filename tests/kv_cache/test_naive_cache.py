import pytest
import torch
from transformers import AutoConfig

from inference.kv_cache.naive_cache import Cache

MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def config() -> AutoConfig:
    return AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)


def test_cache_starts_empty(config: AutoConfig):
    batch = 2
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads

    cache = Cache(
        tag="test",
        batch=batch,
        head_dim=head_dim,
        num_heads=num_kv_heads,
    )

    assert cache.k_cache.shape == (batch, num_kv_heads, 0, head_dim)
    assert cache.v_cache.shape == (batch, num_kv_heads, 0, head_dim)


def test_cache_update_appends(config: AutoConfig):
    torch.manual_seed(1)

    batch = 1
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    seq_len_1, seq_len_2 = 3, 2

    k1 = torch.randn(batch, num_kv_heads, seq_len_1, head_dim)
    v1 = torch.randn(batch, num_kv_heads, seq_len_1, head_dim)

    cache = Cache(
        tag="test",
        batch=batch,
        head_dim=head_dim,
        num_heads=num_kv_heads,
    )

    k_cache, v_cache = cache.update(k1, v1)

    assert k_cache.shape == (batch, num_kv_heads, seq_len_1, head_dim)
    assert v_cache.shape == (batch, num_kv_heads, seq_len_1, head_dim)
    assert torch.allclose(k_cache, k1)
    assert torch.allclose(v_cache, v1)

    k2 = torch.randn(batch, num_kv_heads, seq_len_2, head_dim)
    v2 = torch.randn(batch, num_kv_heads, seq_len_2, head_dim)

    k_cache, v_cache = cache.update(k2, v2)

    expected_k = torch.cat([k1, k2], dim=-2)
    expected_v = torch.cat([v1, v2], dim=-2)

    assert k_cache.shape == (batch, num_kv_heads, seq_len_1 + seq_len_2, head_dim)
    assert v_cache.shape == (batch, num_kv_heads, seq_len_1 + seq_len_2, head_dim)
    assert torch.allclose(k_cache, expected_k)
    assert torch.allclose(v_cache, expected_v)
