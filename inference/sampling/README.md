# Sampling

## Top K
Naive algorithm -> sort the logits from high to low, find the top-k indices, and set the rest to -inf. The bottleneck here is torch.argsort(...), which has O(V logV).

Optimized -> instead of full sort, use heapsort