from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float | None = 1.0
    top_k: int | None = None  # 20
    top_p: float | None = None

    def __post_init__(self):

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"`top_k` must be greater than 0, got {self.top_k}")

        if self.top_p is not None and not (0.0 < float(self.top_p) <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
