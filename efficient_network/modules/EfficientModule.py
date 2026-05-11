import torch
from .FusedMIBModule import FusedMIBModule
from .MIBModule import MIBModule
from .StochasticDepthModule import StochasticDepthModule


class EfficientModule(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, expand_ratio, stride, is_fused, drop_prob
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        if is_fused:
            self.core_block = FusedMIBModule(
                in_channels, out_channels, expand_ratio, stride
            )
        else:
            self.core_block = MIBModule(in_channels, out_channels, expand_ratio, stride)
        self.stochastic_depth = StochasticDepthModule(drop_prob)

    def forward(self, x):
        if self.use_residual:
            return x + self.stochastic_depth(self.core_block(x))
        return self.core_block(x)
