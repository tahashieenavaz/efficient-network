import torch
from .modules import StochasticDepthModule
from .modules import SqueezeExcitationModule
from .modules import ConvolutionalModule
from .EfficientNetworkClassifier import EfficientNetworkClassifier
from .EfficientNetworkPool import EfficientNetworkPool


class EfficientNetwork(torch.nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2):
        super().__init__()
        cfg = [
            (2, 24, 24, 1, 1, True),
            (4, 24, 48, 4, 2, True),
            (4, 48, 64, 4, 2, True),
            (6, 64, 128, 4, 2, False),
            (9, 128, 160, 6, 1, False),
            (15, 160, 256, 6, 2, False),
        ]

        self.features = torch.nn.Sequential(*feats)
        self.pool = EfficientNetworkPool()
        self.classifier = EfficientNetworkClassifier(
            num_classes=num_classes, dropout=dropout, feature_dimension=1280
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
