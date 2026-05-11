import torch


class EfficientNetworkPool(torch.nn.AdaptiveAvgPool2d):
    def __init__(self):
        super().__init__(output_size=1)
