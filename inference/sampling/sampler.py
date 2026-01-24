import torch
from torch import nn
from inference.sampling.sampling_params import SamplingParams


class Sampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        use_lib: bool = True,
    ):
        top_k, top_p = params.top_k, params.top_p
        if params.temperature == 0 or (top_k is None and top_p is None):
            # greedy decoding
            return logits.argmax(dim=-1)

        logits = logits.div(params.temperature)

        if top_k is not None:
            logits = top_k_filter(
                k=top_k,
                logits=logits,
            )

        if top_p is not None:
            logits = top_p_filter(p=top_p, logits=logits)

        probs = torch.softmax(logits, dim=-1)

        if use_lib:
            next_ids = torch.multinomial(probs, num_samples=1)
        else:
            next_ids = sample_multinomial(probs)

        return next_ids


def sample_multinomial(probs: torch.Tensor):
    rr = torch.rand(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
    cumsum = torch.cumsum(probs, dim=-1)

    idx = (cumsum <= rr).sum(dim=-1, keepdim=True)
    return idx


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
        top_k_indices = torch.topk(logits, k, dim=-1).indices
    else:
        top_k_indices = top_k(logits=logits, k=k, dim=-1)

    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask.scatter_(dim=-1, index=top_k_indices, value=True)

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
