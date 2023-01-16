"""CNN-based recognition and alignment models."""

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

from warnings import warn

from typing import Callable


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


def create_resnet(
    resnet_type: int, headless: bool = True, pretrained: bool = True
) -> nn.Module:
    """Generate a ResNet from TorchVision given the ResNet type as integer."""
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
) -> nn.Module:
    """Generate a VGG from TorchVision given the VGG type as integer."""
    vgg_dict = VGGS_BN if batchnorm else VGGS

    vgg = vgg_dict[vgg_type]
    vgg = vgg(pretrained=pretrained)

    if headless:
        vgg = nn.Sequential(*list(vgg.children())[:-2])

    return vgg


class FullyConvCTC(nn.Module):
    """A fully convolutional CTC model with convolutional upsampling."""

    def __init__(
        self,
        width_upsampling: int,
        kern_upsampling: int,
        intermediate_units: int,
        output_units: int,
        resnet_type: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self._backbone = create_resnet(resnet_type, pretrained=pretrained)
        self._pooling = nn.AdaptiveAvgPool2d((1, 32))

        self._upsample = nn.ConvTranspose2d(
            in_channels=RESNET_EMBEDDING_SIZES[resnet_type],
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
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        :param x: Batch of input tensor images.
        """
        # x: N x 3   x H       x W
        x = self._backbone(x)  # N x 512 x H // 32 x W // 32
        x = self._pooling(x)  # N x 512 x 1       x W // 32
        x = self._upsample(x)  # N x INT x 1       x ~(W // 32 - 1) * K
        x = self._activation(x)
        x = self._output(x)  # N x CLS x~(W // 32 - 1) * K
        x = x.squeeze(2)  # N x INT x ~(W // 32 - 1) * K
        x = x.permute((2, 0, 1))  # ~(W // 32 - 1) * K x N x  CLS
        y = self._softmax(x)

        return y

    def load_weights(self, wpath: str) -> None:
        weights = torch.load(wpath)
        missing, unexpected = self.load_state_dict(weights, strict=False)

        if missing or unexpected:
            warn("Careful: Not all weights have been loaded on the model")
            print("Missing: ", missing)
            print("Unexpected: ", unexpected)

    def save_weights(self, path: str) -> None:
        state_dict = self.state_dict()
        torch.save(state_dict, path)


class CustomCNN(nn.Module):
    """Based on Arnau Baró's CRNN CTC model."""

    def __init__(self):
        # Convolution 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        nn.init.xavier_uniform_(self.conv1.weight)  # Xaviers Initialisation
        self.swish1 = nn.ReLU()
        self.conv2_bn1 = nn.BatchNorm2d(32)

        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)

        # Convolution 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        self.swish2 = nn.ReLU()
        self.conv2_bn2 = nn.BatchNorm2d(64)

        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv3.weight)
        self.swish3 = nn.ReLU()
        self.conv2_bn3 = nn.BatchNorm2d(128)

        # Max Pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)

        # Convolution 4
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        nn.init.xavier_uniform_(self.conv4.weight)
        self.swish4 = nn.ReLU()
        self.conv2_bn4 = nn.BatchNorm2d(256)

        # Max Pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv1(x)
        out = self.swish1(out)
        out = self.conv2_bn1(out)
        out, indices1 = self.maxpool1(out)

        out = self.conv2(out)
        out = self.swish2(out)
        out = self.conv2_bn2(out)
        out, indices2 = self.maxpool2(out)

        out = self.conv3(out)
        out = self.swish3(out)
        out = self.conv2_bn3(out)
        out, indices3 = self.maxpool3(out)

        out = self.conv4(out)
        out = self.swish4(out)
        out = self.conv2_bn4(out)
        # out,indices4=self.maxpool4(out)

        if self.training:
            out = nn.functional.dropout(out, 0.5)

        return out
