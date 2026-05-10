import torch
from typing import Type


class SqueezeExcitationModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.scaler = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            activation(inplace=True),
            torch.nn.Conv2d(squeeze_channels, in_channels, kernel_size=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaler(x)
