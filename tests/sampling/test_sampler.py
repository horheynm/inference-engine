import pytest
import torch

from inference.sampling.sampler import top_k

BATCH = 4
VOCAB_SIZE = 1024
TOP_K = 5


@pytest.fixture()
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((BATCH, VOCAB_SIZE), dtype=torch.float32)


def test_top_k(logits: torch.Tensor) -> None:
    top_k_indices = top_k(top_k=TOP_K, logits=logits.clone())
    assert top_k_indices.shape == (BATCH, TOP_K)

    torch_topk_indices = torch.topk(logits, TOP_K, dim=-1).indices
    assert torch.allclose(top_k_indices, torch_topk_indices, atol=0)
