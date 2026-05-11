import torch
from .EfficientNetworkEncoder import EfficientNetworkEncoder
from .EfficientNetworkClassifier import EfficientNetworkClassifier
from .EfficientNetworkPool import EfficientNetworkPool


class EfficientNetwork(torch.nn.Module):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        dropout: float = 0.2,
        feature_dimension: int = 1280,
        channels: int = 3,
    ):
        super().__init__()
        assert variant in ["small", "medium", "large", "s", "m", "l"]
        self.encoder = EfficientNetworkEncoder(
            variant=variant, feature_dimension=feature_dimension, in_channels=channels
        )
        self.pool = EfficientNetworkPool()
        self.classifier = EfficientNetworkClassifier(
            num_classes=num_classes,
            dropout=dropout,
            feature_dimension=feature_dimension,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.pool(features)
        return self.classifier(features)
