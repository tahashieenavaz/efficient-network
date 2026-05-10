import torch


class EfficientNetworkClassifier(torch.nn.Module):
    def __init__(self, *, dropout: float, in_features: int, num_classes: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout, inplace=True)
        self.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        return self.fc(features)
