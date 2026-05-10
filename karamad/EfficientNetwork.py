import torch
import torch.nn as nn


def ConvBlock(in_c, out_c, kernel_size=3, stride=1, groups=1, act=True):
    layers = [
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.1),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class EfficientBlock(nn.Module):
    def __init__(self, in_c, out_c, exp, stride, fused, stoch_p):
        super().__init__()
        self.res = stride == 1 and in_c == out_c
        hid_c = in_c * exp

        if fused:
            if exp != 1:
                layers = [
                    ConvBlock(in_c, hid_c, 3, stride),
                    ConvBlock(hid_c, out_c, 1, act=False),
                ]
            else:
                layers = [ConvBlock(in_c, out_c, 3, stride)]
        else:
            layers = []
            if exp != 1:
                layers.append(ConvBlock(in_c, hid_c, 1))
            layers.extend(
                [
                    ConvBlock(hid_c, hid_c, 3, stride, groups=hid_c),
                    SqueezeExcitation(hid_c, max(1, in_c // 4)),
                    ConvBlock(hid_c, out_c, 1, act=False),
                ]
            )

        self.block = nn.Sequential(*layers)
        self.drop = StochasticDepth(stoch_p)

    def forward(self, x):
        return x + self.drop(self.block(x)) if self.res else self.block(x)


class EfficientNetV2_S(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Config: (repeats, in_c, out_c, expand_ratio, stride, is_fused)
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

        for repeats, in_c, out_c, exp, stride, fused in cfg:
            for i in range(repeats):
                feats.append(
                    EfficientBlock(
                        in_c if i == 0 else out_c,
                        out_c,
                        exp,
                        stride if i == 0 else 1,
                        fused,
                        0.005 * b_idx,
                    )
                )
                b_idx += 1

        feats.append(ConvBlock(256, 1280, 1))

        self.features = nn.Sequential(*feats)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
