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
            top_k_indices = top_k(
                top_k=top_k,
                logits=logits,
            )

            logits = logits.gather(index=top_k_indices, dim=-1)

        if (top_p := params.top_p) is not None:
            top_p_indices = top_p(top_p=top_p, logits=logits)
            logits = logits.gather(index=top_p_indices, dim=-1)


def top_k(
    top_k: int,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Return the top_k indices
    """
    top_k = min(top_k, logits.size(-1))
    indices = torch.argsort(
        logits,
        descending=True,
        dim=-1,
    )
    return indices[..., :top_k]


def top_p(
    top_p: float,
    logits: torch.Tensor,
):
    """
    Return the top_p indices
    """
    logit_probs = torch.softmax(logits, dim=-1)
    # cumsum = torch.cumsum(logit_probs, dim=-1)
    sorted_idx = torch.argsort(logits, descending=True, dim=-1)
    sorted_probs = logit_probs.gather(index=sorted_idx, dim=-1)
    cdf = sorted_probs.cumsum(dim=-1)

    # indices to keep
    mask = cdf < top_p

    # shift to include probs that include 0.8
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True

    filtered_sorted_probs = sorted_probs * mask
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(
        dim=-1, keepdim=True
    )

    sampled_sorted = torch.multinomial(filtered_sorted_probs, 1)  # (...)

    next_token_index = sorted_idx.gather(-1, sampled_sorted)

    return next_token_index
