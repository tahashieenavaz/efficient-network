import torch
from .EfficientNetworkEncoder import EfficientNetworkEncoder
from .EfficientNetworkClassifier import EfficientNetworkClassifier
from .EfficientNetworkPool import EfficientNetworkPool


class EfficientNetwork(torch.nn.Module):
    def __init__(self, version: str, num_classes: int, dropout: float = 0.2):
        super().__init__()
        assert version in ["small", "medium", "large", "s", "m", "l"]
        self.encoder = EfficientNetworkEncoder()
        self.pool = EfficientNetworkPool()
        self.classifier = EfficientNetworkClassifier(
            num_classes=num_classes, dropout=dropout, feature_dimension=1280
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.pool(features)
        return self.classifier(features)
