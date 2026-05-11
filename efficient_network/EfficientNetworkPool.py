import torch


class EfficientNetworkPool(torch.nn.Module):
    def __init__(self):
        self.stream = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
