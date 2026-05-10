# Karamad

PyTorch implementation of EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan and Quoc V. Le.

## Installation

```bash
pip install karamad
```

## Usage

```py
from karamad import EfficientNet

model = EfficientNet()

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