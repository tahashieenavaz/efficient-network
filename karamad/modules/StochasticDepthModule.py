import torch


class StochasticDepthModule(torch.nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_probability = 1.0 - self.probability
        mask = (torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep_probability).to(
            x.dtype
        )
        return x * mask / keep_probability
