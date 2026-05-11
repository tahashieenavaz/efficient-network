# Efficient Networks V2 Implementation

PyTorch implementation of EfficientNetV2: Smaller Models and Faster Training by Mingxing Tan and Quoc V. Le.

## Installation

```bash
pip install efficient-network
```

## Usage

```py
from efficient_network import EfficientNetwork

model = EfficientNetwork()

training_loop(model)
```

## Citation

```bibtex
@article{1905.11946,
    Title = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
    Author = {Mingxing Tan and Quoc V. Le},
    Year = {2019},
    Eprint = {arXiv:1905.11946},
    Howpublished = {International Conference on Machine Learning, 2019},
}
```