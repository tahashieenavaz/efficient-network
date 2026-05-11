import torch
from .EfficientNetworkConfig import EfficientNetworkConfig
from .modules import ConvolutionalModule
from .modules import EfficientModule


class EfficientNetworkEncoder(torch.nn.Module):
    def __init__(self, variant: str, feature_dimension: int, in_channels: int):
        super().__init__()
        config = EfficientNetworkConfig(variant).get()

        layers = [
            ConvolutionalModule(
                in_channels=in_channels, out_channels=24, kernel_size=3, stride=2
            )
        ]

        block_index = 0
        for repeats, in_c, out_c, exp_ratio, stride, is_fused in config:
            for i in range(repeats):
                current_stride = stride if i == 0 else 1
                current_in_channels = in_c if i == 0 else out_c
                drop_prob = 0.005 * block_index
                layers.append(
                    EfficientModule(
                        in_channels=current_in_channels,
                        out_channels=out_c,
                        expand_ratio=exp_ratio,
                        stride=current_stride,
                        is_fused=is_fused,
                        drop_prob=drop_prob,
                    )
                )
                block_index += 1

        layers.append(
            ConvolutionalModule(
                in_channels=256, out_channels=feature_dimension, kernel_size=1, stride=1
            )
        )
        self.stream = torch.nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.stream(images)
