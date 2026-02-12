import pytest
import torch
from transformers import AutoConfig

from inference.layers.attention.naive_attention import Attention
from inference.layers.attention.sdpa_attention import scaled_dot_product_attention

MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def config() -> AutoConfig:
    return AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def attention_params(config: AutoConfig):
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    scale = head_dim**-0.5
    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "scale": scale,
    }


def test_prefill_matches_sdpa(attention_params):
    torch.manual_seed(1)

    num_heads = attention_params["num_heads"]
    num_kv_heads = attention_params["num_kv_heads"]
    head_dim = attention_params["head_dim"]
    scale = attention_params["scale"]

    batch, seq_len = 2, 4
    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim)

    attention = Attention(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        layer_idx=0,
        scale=scale,
    )

    actual = attention(q, k, v)
    expected = scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        is_causal=True,
        scale=scale,
        enable_gqa=(num_heads != num_kv_heads),
    ).transpose(1, 2)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cache_appends_and_is_used(attention_params):
    num_heads = attention_params["num_heads"]
    num_kv_heads = attention_params["num_kv_heads"]
    head_dim = attention_params["head_dim"]
    scale = attention_params["scale"]

    batch, seq_len, next_seq_len = 1, 3, 2
    torch.manual_seed(1)
    q1 = torch.randn(batch, seq_len, num_heads, head_dim)
    k1 = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    v1 = torch.randn(batch, seq_len, num_kv_heads, head_dim)

    attention = Attention(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        layer_idx=0,
        scale=scale,
    )

    _ = attention(q1, k1, v1)
    assert attention.cache.k_cache.shape == (batch, num_kv_heads, seq_len, head_dim)
    assert attention.cache.v_cache.shape == (batch, num_kv_heads, seq_len, head_dim)

    q2 = torch.randn(batch, next_seq_len, num_heads, head_dim)
    k2 = torch.randn(batch, next_seq_len, num_kv_heads, head_dim)
    v2 = torch.randn(batch, next_seq_len, num_kv_heads, head_dim)

    actual = attention(q2, k2, v2)

    assert attention.cache.k_cache.shape == (
        batch,
        num_kv_heads,
        seq_len + next_seq_len,
        head_dim,
    )
    assert attention.cache.v_cache.shape == (
        batch,
        num_kv_heads,
        seq_len + next_seq_len,
        head_dim,
    )

    expected = scaled_dot_product_attention(
        query=q2.transpose(1, 2),
        key=torch.cat([k1, k2], dim=1).transpose(1, 2),
        value=torch.cat([v1, v2], dim=1).transpose(1, 2),
        is_causal=True,
        scale=scale,
        enable_gqa=(num_heads != num_kv_heads),
    ).transpose(1, 2)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
