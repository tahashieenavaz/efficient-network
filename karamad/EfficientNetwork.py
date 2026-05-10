import torch
from .EfficientNetworkEncoder import EfficientNetworkEncoder
from .EfficientNetworkClassifier import EfficientNetworkClassifier
from .EfficientNetworkPool import EfficientNetworkPool


class EfficientNetwork(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        classifier_dropout: float = 0.2,
        classifier_in_features: int = 1280,
    ):
        super().__init__()
        self.encoder = EfficientNetworkEncoder()
        self.pool = EfficientNetworkPool()
        self.classifier = EfficientNetworkClassifier(
            num_classes=num_classes,
            dropout=classifier_dropout,
            in_features=classifier_in_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)
