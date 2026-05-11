import torch
from .ConvolutionalModule import ConvolutionalModule
from .SqueezeExcitationModule import SqueezeExcitationModule


class MIBModule(torch.nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, expansion: int, stride: int
    ):
        hidden_channels = in_channels * expansion
        layers = []

        if expansion != 1:
            layers.append(
                ConvolutionalModule(in_channels, hidden_channels, kernel_size=1)
            )

        layers.extend(
            [
                ConvolutionalModule(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                ),
                SqueezeExcitationModule(
                    hidden_channels, squeeze_channels=max(1, in_channels // 4)
                ),
                ConvolutionalModule(
                    hidden_channels, out_channels, kernel_size=1, use_activation=False
                ),
            ]
        )

        super().__init__(*layers)
