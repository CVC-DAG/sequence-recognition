import torch
from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()