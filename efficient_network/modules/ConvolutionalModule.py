import torch


class ConvolutionalModule(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        groups: int = 1,
        use_activation: bool = True,
        momentum: float = 0.1,
        epsilon: float = 0.001,
    ):
        padding = kernel_size // 2
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, eps=epsilon, momentum=momentum),
        ]
        if use_activation:
            layers.append(torch.nn.SiLU(inplace=True))
        super().__init__(*layers)
