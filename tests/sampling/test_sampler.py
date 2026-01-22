import pytest
import torch

from inference.sampling.sampler import top_k


@pytest.fixture()
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    # (batch, vocab)
    return torch.randn(4, 17, dtype=torch.float32)


def test_top_k(logits: torch.Tensor) -> None:
    k = 5
    processed = top_k(top_k=k, logits=logits.clone())

    assert processed.shape == logits.shape

    # Compare against torch.topk: the only finite positions should be exactly the top-k indices.
    expected_topk = torch.topk(logits, k, dim=-1)
    expected_mask = torch.zeros_like(logits, dtype=torch.bool)
    expected_mask.scatter_(dim=-1, index=expected_topk.indices, value=True)
    assert torch.equal(torch.isfinite(processed), expected_mask)

    # Values at kept indices should be unchanged; all others should be -inf.
    assert torch.equal(processed[expected_mask], logits[expected_mask])
    assert torch.isneginf(processed[~expected_mask]).all()

    # top_k=1 behaves like greedy: only the argmax token remains finite.
    processed_k1 = top_k(top_k=1, logits=logits.clone())
    argmax_idx = logits.argmax(dim=-1)

    for b in range(logits.shape[0]):
        kept = processed_k1[b, argmax_idx[b]].item()
        original = logits[b, argmax_idx[b]].item()
        assert kept == original

        # Everything else should be filtered out.
        mask = torch.ones(logits.shape[1], dtype=torch.bool)
        mask[argmax_idx[b]] = False
        assert torch.isneginf(processed_k1[b, mask]).all()
