import torch
from typing import Type


class SqueezeExcitationModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
        scale_activation: Type[torch.nn.Module] = torch.nn.Sigmoid,
    ):
        super().__init__()
        squeeze_channels = max(1, in_channels // 4)
        self.alpha = torch.nn.Conv2d(expanded_channels, squeeze_channels, kernel_size=1)
        self.beta = torch.nn.Conv2d(squeeze_channels, expanded_channels, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def __get_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.alpha(scale)
        scale = self.activation(scale)
        scale = self.beta(scale)
        scale = self.scale_activation(scale)
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.__get_scale(x)
        return scale * x
