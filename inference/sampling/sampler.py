import torch
from torch import nn
from inference.sampling.sampling_params import SamplingParams

# TODO: Use flashinfer kernels

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
        use_lib: bool = False,
    ):
        temperature = params.temperature
        top_k = params.top_k
        top_p = params.top_p

        if params.temperature is None or params.temperature < 0:
            # greedy decoding
            return logits.argmax(dim=-1, keepdim=True)

        if temperature is not None:
            logits = logits.div(temperature)

        if top_k is not None:
            logits = apply_top_k_logits(
                k=top_k,
                logits=logits,
            )

        if top_p is not None:
            logits = apply_top_p_logits(p=top_p, logits=logits)

        probs = torch.softmax(logits.float(), dim=-1)  # fp32

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

    indices = torch.argsort(logits, descending=True, dim=dim)  # O(V logV)
    top_k_indices = indices[..., :k]
    return top_k_indices


def top_k_heap(logits: torch.Tensor, k: int):
    """
    Algorithmically better top k O(B V log k), but CPU heavy.
    Implemented for conceptual purposes
    """
    import heapq

    B, V = logits.shape
    k = min(k, V)

    out = torch.empty((logits.shape[0], k), dtype=torch.int64)
    for b in range(B):
        row = logits[b]

        heap = []
        for i in range(k):
            heapq.heappush(heap, ((row[i]), i))

        for i in range(k, V):
            val = float(row[i])
            if val > heap[0][0]:
                heapq.heapreplace(heap, (val, i))

        heap.sort(reverse=True)
        out[b] = torch.tensor([idx for _, idx in heap])

    return out


def apply_top_k_logits(
    logits: torch.Tensor,
    k: int,
    filter_value: float = -float("inf"),
    use_lib: bool = True,
) -> torch.Tensor:
    """
    Keep only top-k logits per row; mask the rest to -inf
    logits: [..., vocab]

    """
    if k < 0:
        raise ValueError(f"k must be greater than 0, got {k}")

    vocab = logits.size(-1)
    k = min(k, vocab)

    if use_lib:
        top_k_indices = torch.topk(logits, k, dim=-1).indices
    else:
        top_k_indices = top_k(logits=logits, k=k, dim=-1)  # O(V logV)
        # top_k_indices = top_k_heap(logits=logits, k=k) # O(V logk), but not GPU friendly

    filter_idx = torch.ones_like(logits, dtype=torch.bool)
    filter_idx.scatter_(dim=-1, index=top_k_indices, value=False)

    return logits.masked_fill(mask=filter_idx, value=filter_value)


def apply_top_p_logits(
    logits: torch.Tensor,
    p: float,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Keep only top-p logits per row; mask the rest to -inf.
    logits: [..., vocab]

    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)

    m = sorted_logits[..., 0, None]
    probs_sorted = (sorted_logits - m).softmax(dim=-1)
    cdf = probs_sorted.cumsum(dim=-1)

    # remove tokens strictly after the nucleus; keep the token that crosses p
    to_remove_sorted = cdf > p
    to_remove_sorted = to_remove_sorted.roll(shifts=1, dims=-1)
    to_remove_sorted[..., 0] = False

    if min_tokens_to_keep > 1:
        to_remove_sorted[..., :min_tokens_to_keep] = False

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=to_remove_sorted)

    return logits.masked_fill(mask=mask, value=filter_value)
