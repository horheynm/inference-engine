import pytest
import torch
from transformers import AutoConfig

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


def maybe_gqa_interleave(
    x: torch.Tensor, num_heads: int, num_kv_heads: int
) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return x
    repeat = num_heads // num_kv_heads
    return x.repeat_interleave(repeat, dim=1)


def test_matches_torch_sdpa(attention_params):
    torch.manual_seed(1)

    num_heads = attention_params["num_heads"]
    num_kv_heads = attention_params["num_kv_heads"]
    head_dim = attention_params["head_dim"]
    scale = attention_params["scale"]

    batch, seq_len = 2, 3
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim)

    enable_gqa = num_heads != num_kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        maybe_gqa_interleave(k, num_heads, num_kv_heads),
        maybe_gqa_interleave(v, num_heads, num_kv_heads),
        is_causal=True,
        dropout_p=0.0,
        scale=scale,
    )

    actual = scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        is_causal=True,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_attn_mask_respected(attention_params):
    num_heads = attention_params["num_heads"]
    num_kv_heads = attention_params["num_kv_heads"]
    head_dim = attention_params["head_dim"]
    scale = attention_params["scale"]

    batch, seq_len = 1, 2
    # simple tensors make it easy to check masking behavior
    q = torch.ones(batch, num_heads, seq_len, head_dim)
    k = torch.ones(batch, num_kv_heads, seq_len, head_dim)
    v = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
    v = v.repeat(batch, num_kv_heads, 1, head_dim)

    # mask out attention to the last token
    attn_mask = torch.tensor(
        [
            [True, False],
            [True, False],
        ],
        dtype=torch.bool,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        maybe_gqa_interleave(k, num_heads, num_kv_heads),
        maybe_gqa_interleave(v, num_heads, num_kv_heads),
        attn_mask=attn_mask,
        is_causal=False,
        dropout_p=0.0,
        scale=scale,
    )

    actual = scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        is_causal=False,
        scale=scale,
        enable_gqa=(num_heads != num_kv_heads),
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
