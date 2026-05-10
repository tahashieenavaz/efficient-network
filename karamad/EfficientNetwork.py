import torch
from .modules import StochasticDepthModule
from .modules import SqueezeExcitationModule
from .modules import ConvolutionalModule
from .EfficientNetworkClassifier import EfficientNetworkClassifier
from .EfficientNetworkPool import EfficientNetworkPool


class EfficientBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride,
        fused,
        stoch_p,
    ):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion

        if fused:
            if expansion != 1:
                layers = [
                    ConvolutionalModule(in_channels, expanded_channels, 3, stride),
                    ConvBlock(expanded_channels, out_channels, 1, act=False),
                ]
            else:
                layers = [ConvBlock(in_channels, out_channels, 3, stride)]
        else:
            layers = []
            if expansion != 1:
                layers.append(ConvBlock(in_channels, expanded_channels, 1))
            layers.extend(
                [
                    ConvBlock(
                        expanded_channels,
                        expanded_channels,
                        3,
                        stride,
                        groups=expanded_channels,
                    ),
                    SqueezeExcitationModule(
                        expanded_channels, max(1, in_channels // 4)
                    ),
                    ConvBlock(expanded_channels, out_channels, 1, act=False),
                ]
            )

        self.block = torch.nn.Sequential(*layers)
        self.drop = StochasticDepthModule(stoch_p)

    def forward(self, x):
        return x + self.drop(self.block(x)) if self.res else self.block(x)


class EfficientNetV2_S(torch.nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # Config: (repeats, in_channels, out_channels, expand_ratio, stride, is_fused)
        cfg = [
            (2, 24, 24, 1, 1, True),
            (4, 24, 48, 4, 2, True),
            (4, 48, 64, 4, 2, True),
            (6, 64, 128, 4, 2, False),
            (9, 128, 160, 6, 1, False),
            (15, 160, 256, 6, 2, False),
        ]

        feats = [ConvBlock(3, 24, 3, 2)]
        b_idx = 0

        for repeats, in_channels, out_channels, exp, stride, fused in cfg:
            for i in range(repeats):
                feats.append(
                    EfficientBlock(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        exp,
                        stride if i == 0 else 1,
                        fused,
                        0.005 * b_idx,
                    )
                )
                b_idx += 1

        feats.append(ConvBlock(256, 1280, 1))

        self.features = torch.nn.Sequential(*feats)
        self.avgpool = EfficientNetworkPool()
        self.classifier = EfficientNetworkClassifier()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
