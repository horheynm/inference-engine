import pytest
import torch

from inference.sampling.sampler import top_k, top_k_filter, top_p_filter

BATCH = 4
VOCAB_SIZE = 10
TOP_K = 5


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

    p = 0.8
    wrapper = TopPLogitsWarper(top_p=p)
    expected = wrapper(input_ids=None, scores=logits)

    filtered_logits = top_p_filter(p=p, logits=logits)
    assert torch.allclose(expected, filtered_logits)
