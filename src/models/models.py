from torchvision.models import resnet18, resnet50

from torch import nn
import torch


class CTCAligner(nn.Module):
    RESNET_TYPE = {
        "resnet18": resnet18,
        "resnet50": resnet50,
    }

    def __init__(
            self,
            rnet_type: str,

    ) -> None:
        super(CTCAligner).__init__()

        self.cnn = self.RESNET_TYPE[rnet_type](pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

        self.rnn = nn.LSTM()


    def forward(
            self,
            
    ):
        ...