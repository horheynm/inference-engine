from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: None | int = None
    top_p: None | float = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be greater than 0")

        if self.top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {self.top_k}"
            )
