import torch


class EfficientNetworkEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = EfficientNetworkConfig(variant).get()

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass
