import torch
from torch import nn

from data.generic_decrypt import GenericSample

class FullyConvCTC(nn.Module):
    def __init__(
                self,
        ) -> None:
        super(FullyConvCTC).__init__()

        self.conv1 = nn.Conv2d()
    
    def forward(
            self,
            x: GenericSample
    ) -> torch.Tensor:
        ...