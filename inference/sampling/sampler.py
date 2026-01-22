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

        if params.top_k is not None:
            logits = top_k(
                logits=logits,
            )


def top_k(
    top_k: int,
    logits: torch.Tensor,
    min_tokens_to_keep: int = 1,
    filter_value: float = float("-inf"),
):
    top_k = min(top_k, logits.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    return scores_processed
