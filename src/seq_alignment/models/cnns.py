"""CNN utilities."""

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from torchvision.models import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)


RESNETS = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152,
}

RESNET_EMBEDDING_SIZES = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}

VGGS = {
    11: vgg11,
    13: vgg13,
    16: vgg16,
    19: vgg19,
}

VGGS_BN = {
    11: vgg11_bn,
    13: vgg13_bn,
    16: vgg16_bn,
    19: vgg19_bn,
}

VGG_EMBEDDING_SIZE = 512


def create_resnet(
    resnet_type: int, headless: bool = True, pretrained: bool = True
) -> nn.Module:
    """Generate a ResNet from TorchVision given the ResNet type as integer.

    Parameters
    ----------
    resnet_type: int
        The number that represents the ResNet type (18, 34, 50, 101, 152).
    headless: bool
        Whether or not to remove the classification and embedding layers atop
        the default ResNet.
    pretrained: bool
        Whether to use pretrained weights or not.

    Returns
    -------
    nn.Module
        A ResNet model ready for inference.
    """
    resnet = RESNETS[resnet_type]
    resnet = resnet(pretrained=pretrained)

    if headless:
        resnet = nn.Sequential(*list(resnet.children())[:-2])
    return resnet


def create_vgg(
    vgg_type: int,
    batchnorm: bool = True,
    headless: bool = True,
    pretrained: bool = True,
    keep_maxpool: bool = True,
) -> nn.Module:
    """Generate a VGG from TorchVision given the VGG type as integer.

    Parameters
    ----------
    resnet_type: int
        The number that represents the VGG type (11, 13, 16, 19).
    headless: bool
        Whether or not to remove the classification and embedding layers atop
        the default VGG.
    pretrained: bool
        Whether to use pretrained weights or not.

    Returns
    -------
    nn.Module
        A VGG model ready for inference.
    """
    vgg_dict = VGGS_BN if batchnorm else VGGS

    vgg = vgg_dict[vgg_type]
    vgg = vgg(pretrained=pretrained)

    if headless:
        vgg = nn.Sequential(*list(vgg.children())[:-2])

        if not keep_maxpool:
            vgg[0] = nn.Sequential(*list(vgg[0].children())[:-1])

    return vgg


class BaroCNN(nn.Module):
    """ConvNet model based on Arnau Baró's CRNN CTC."""

    def __init__(
        self,
        dropout: float = 0.1,
    ) -> None:
        """Initialise BaroCNN model.

        Parameters
        ----------
        dropout: float
            Dropout to be applied to the final embedding.
        """
        super().__init__()

        # Convolution 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        nn.init.xavier_uniform_(self.conv1.weight)  # Xaviers Initialisation
        self.swish1 = nn.ReLU()
        self.conv2_bn1 = nn.BatchNorm2d(32)

        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Convolution 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        self.swish2 = nn.ReLU()
        self.conv2_bn2 = nn.BatchNorm2d(64)

        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Convolution 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv3.weight)
        self.swish3 = nn.ReLU()
        self.conv2_bn3 = nn.BatchNorm2d(128)

        # Max Pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1))

        # Convolution 4
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv4.weight)
        self.swish4 = nn.ReLU()
        self.conv2_bn4 = nn.BatchNorm2d(256)

        # Max Pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))

        self.dropout = nn.Dropout()

    def forward(self, x):
        """Perform inference.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        out = self.conv1(x)
        out = self.swish1(out)
        out = self.conv2_bn1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.swish2(out)
        out = self.conv2_bn2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.swish3(out)
        out = self.conv2_bn3(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.swish4(out)
        out = self.conv2_bn4(out)
        out = self.maxpool4(out)

        out = self.avg_pool(out)
        out = self.dropout(out)

        out = out.squeeze(2)  # Remove height dimension

        return out  # Batch x Features x Width
