import pytest
import torch

from inference.sampling.sampler import top_k, top_k_filter, top_p_filter, Sampler
from inference.sampling.sampling_params import SamplingParams

BATCH = 4
VOCAB_SIZE = 1042
TOP_K = 20
TOP_P = 0.9


@pytest.fixture()
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((BATCH, VOCAB_SIZE), dtype=torch.float32)


def test_top_k(logits: torch.Tensor) -> None:
    top_k_indices = top_k(k=TOP_K, logits=logits.clone())
    assert top_k_indices.shape == (BATCH, TOP_K)

    torch_topk_indices = torch.topk(logits, TOP_K, dim=-1).indices
    assert torch.allclose(top_k_indices, torch_topk_indices, atol=0)


@pytest.mark.parametrize("use_lib", (True, False))
def test_top_k_filter(logits: torch.Tensor, use_lib: bool) -> None:
    filter_value = float("-inf")
    filtered_logits = top_k_filter(
        logits=logits, k=TOP_K, filter_value=filter_value, use_lib=use_lib
    )
    indices = torch.topk(logits, TOP_K, dim=-1).indices

    thresh = logits.gather(index=indices[..., -1, None], dim=-1)
    remove_mask = thresh > logits

    expected = logits.masked_fill(remove_mask, filter_value)
    assert torch.allclose(expected, filtered_logits, atol=0)


def test_top_p_filter(logits: torch.Tensor):
    from transformers.generation import TopPLogitsWarper

    wrapper = TopPLogitsWarper(top_p=TOP_P)
    expected = wrapper(input_ids=None, scores=logits)

    filtered_logits = top_p_filter(p=TOP_P, logits=logits)
    assert torch.allclose(expected, filtered_logits)


@pytest.mark.parametrize("use_lib", (True, False))
def test_sampler(logits: torch.Tensor, use_lib: bool) -> None:

    torch.manual_seed(0)

    params = SamplingParams(top_k=TOP_K, top_p=TOP_P, temperature=1.0)

    sampler = Sampler()
    next_ids = sampler(logits=logits, params=params, use_lib=use_lib)

    assert next_ids.shape == (BATCH, 1)

    filtered_logits = logits.div(params.temperature)
    filtered_logits = top_k_filter(logits=filtered_logits, k=TOP_K)
    filtered_logits = top_p_filter(logits=filtered_logits, p=TOP_P)

    chosen_logits = filtered_logits.gather(index=next_ids, dim=-1).squeeze(-1)
    assert torch.isfinite(chosen_logits).all()


def test_sampler_greedy(logits: torch.Tensor) -> None:
    params = SamplingParams(top_k=None, top_p=None, temperature=1.0)
    sampler = Sampler()

    next_ids = sampler(logits=logits, params=params).squeeze(-1)
    expected = logits.argmax(dim=-1)
    assert torch.equal(next_ids, expected)
