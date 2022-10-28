import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class FullyConvCTC(nn.Module):
    RESNETS = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(
                self,
                width_upsampling: int,
                kern_upsampling: int,
                intermediate_units: int,
                output_units: int,
                backbone: callable,
                pretrained: bool = True,
        ) -> None:
        super().__init__()

        backmodel = backbone(pretrained=pretrained)
        self._backbone = nn.Sequential(*list(backmodel.children())[:-2])
        self._pooling = nn.AdaptiveAvgPool2d((1, 32))
        
        self._upsample = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=intermediate_units,
            kernel_size=(1, kern_upsampling),
            stride=(1, width_upsampling),
        )
        self._activation = nn.ReLU()
        self._output = nn.Conv2d(
            kernel_size=1,
            in_channels=intermediate_units,
            out_channels=output_units,
        )
        self._softmax = nn.Softmax(dim=-1)
        

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
                                    # x: N x 3   x H       x W
        x = self._backbone(x)       #    N x 512 x H // 32 x W // 32
        x = self._pooling(x)        #    N x 512 x 1       x W // 32
        x = self._upsample(x)       #    N x INT x 1       x ~(W // 32 - 1) * K
        x = self._activation(x)
        x = self._output(x)         #    N x CLS x~(W // 32 - 1) * K
        x = x.squeeze(2)            #    N x INT x ~(W // 32 - 1) * K
        x = x.permute((0, 2, 1))    #    N x ~(W // 32 - 1) * K x  CLS
        y = self._softmax(x)

        return y
        