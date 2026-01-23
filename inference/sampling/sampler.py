import torch
from torch import nn
from inference.sampling.sampling_params import SamplingParams


class Sampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def foward(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
    ):
        logits = logits.div_(params.temperature)

        if (top_k := params.top_k) is not None:
            logits = top_k_filter(
                top_k=top_k,
                logits=logits,
            )

        if (top_p := params.top_p) is not None:
            top_p_indices = top_p_filter(top_p=top_p, logits=logits)
            logits = logits.gather(index=top_p_indices, dim=-1)

        return logits


def top_k(logits: torch.Tensor, k: int, dim: int = -1):
    """Return the top k indices"""

    indices = torch.argsort(logits, descending=True, dim=dim)
    top_k_indices = indices[..., :k]
    return top_k_indices


def top_k_filter(
    logits: torch.Tensor,
    k: int,
    filter_value: float = -float("inf"),
    use_lib: bool = True,
) -> torch.Tensor:
    """
    Keep only top-k logits per row; mask the rest to -inf.
    logits: [..., vocab]

    """
    vocab = logits.size(-1)
    k = min(k, vocab)

    if use_lib:
        top_k_vals = torch.topk(logits, k, dim=-1).values
    else:
        indices = top_k(logits=logits, k=k, dim=-1)
        top_k_vals = logits.gather(index=indices, dim=-1)

    thresh = top_k_vals[..., -1, None]
    keep_mask = logits >= thresh
    return logits.masked_fill(~keep_mask, filter_value)


def top_p_filter(
    logits: torch.Tensor,
    p: float,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Return the top_p indices
    """
    if not (0.0 < float(p) <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)

    # use max for stability
    m = sorted_logits[..., 0, None]
    probs_sorted = (sorted_logits - m).softmax(dim=-1)
    cdf = probs_sorted.cumsum(dim=-1)

    # remove tokens strictly after the nucleus; keep the token that crosses p
    to_remove_sorted = cdf > p
    to_remove_sorted = to_remove_sorted.roll(shifts=1, dims=-1)
    to_remove_sorted[..., 0] = False

    if min_tokens_to_keep > 1:
        to_remove_sorted[..., :min_tokens_to_keep] = False

    to_remove = torch.zeros_like(logits, dtype=torch.bool)
    to_remove.scatter_(dim=-1, index=sorted_idx, src=to_remove_sorted)

    return logits.masked_fill(to_remove, filter_value)
