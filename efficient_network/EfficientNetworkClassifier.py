import torch


class EfficientNetworkClassifier(torch.nn.Module):
    def __init__(self, *, dropout: float, feature_dimension: int, num_classes: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout, inplace=True)
        self.head = torch.nn.Linear(feature_dimension, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        return self.head(features)
