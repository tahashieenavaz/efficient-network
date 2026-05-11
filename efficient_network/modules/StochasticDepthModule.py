import torch


class StochasticDepthModule(torch.nn.Module):
    def __init__(self, drop_probability: float):
        super().__init__()
        self.drop_probability = drop_probability
        self.survival_rate = 1.0 - self.drop_probability

    def forward(self, x):
        if not self.training or self.drop_probability == 0.0:
            return x

        random_tensor = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        binary_mask = (random_tensor < self.survival_rate).to(x.dtype)
        return x * binary_mask / self.survival_rate
