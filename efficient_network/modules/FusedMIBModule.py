import torch
from .ConvolutionalModule import ConvolutionalModule


class FusedMIBModule(torch.nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, expansion: int, stride: int
    ):
        hidden_channels = in_channels * expansion
        layers = []

        if expansion != 1:
            layers.append(
                ConvolutionalModule(
                    in_channels, hidden_channels, kernel_size=3, stride=stride
                )
            )
            layers.append(
                ConvolutionalModule(
                    hidden_channels, out_channels, kernel_size=1, use_activation=False
                )
            )
        else:
            layers.append(
                ConvolutionalModule(
                    in_channels, out_channels, kernel_size=3, stride=stride
                )
            )

        super().__init__(*layers)
