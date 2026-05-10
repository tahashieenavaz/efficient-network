import torch
import inspect
from typing import Type
from typing import Literal


class ConvolutionalBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        padding: Literal["same", "valid"] | int = "same",
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
        normalization_epsilon: float = 1e-05,
        normalization_momentum: float = 0.1,
    ):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stirde=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.normalization = torch.nn.BatchNorm2d(
            out_channels, eps=normalization_epsilon, momentum=normalization_momentum
        )
        self.__initialize_activation(activation=activation, out_channels=out_channels)

    def __initialize_activation(
        self, *, activation: Type[torch.nn.Module], out_channels: int
    ):
        signature = inspect.signature(activation)
        params = signature.parameters

        if "channels" in params:
            self.activation = activation(channels=out_channels)
        elif "in_channels" in params:
            self.activation = activation(in_channels=out_channels)
        else:
            self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x
