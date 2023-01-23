"""CRNN-based models."""
from __future__ import annotations

import torch
from torch import nn
from warnings import warn

from .base_model import BaseModel
from .cnns import create_resnet, RESNET_EMBEDDING_SIZES, BaroCNN


class BaroCRNN(BaseModel):
    """CRNN Model based on Arnau Baró's CTC OMR model."""

    def __init__(
        self,
        lstm_hidden_size: int,
        lstm_layers: int,
        blstm: bool,
        dropout: float,
        output_classes: int,
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        lstm_hidden_size: int
            Dimensionality within the stack of recurrent layers.
        lstm_layers: int
            Number of recurrent layers to stack.
        blstm: bool
            Whether the stack of recurrent layers will be bidirectional.
        output_classes: int
            Number of output classes in the vocabulary.
        """
        super().__init__()
        self.directions = 2 if blstm else 1

        self.backbone = BaroCNN(dropout)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=blstm,
        )
        self.linear = nn.Linear(lstm_hidden_size, output_classes)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

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
        x = self.backbone(x)
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape

            x = x.view(
                seq_len,
                batch_size,
                self.directions,
                hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.linear(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x


class ResnetCRNN(BaseModel):
    """CRNN Model with a ResNet as backcbone."""

    def __init__(
        self,
        resnet_type: int,
        lstm_layers: int,
        lstm_hidden_size: int,
        upsampling_kern: int,
        upsampling_stride: int,
        blstm: bool,
        dropout: float,
        output_classes: int,
    ) -> None:
        """Initialise ResnetCRNN from parameters.

        Parameters
        ----------
        resnet_type: int
            What type of ResNet to employ as backbone.
        lstm_layers: int
            Number of recurrent layers to stack.
        lstm_hidden_size: int
            Dimensionality within the stack of recurrent layers.
        blstm: bool
            Whether the stack of recurrent layers will be bidirectional.
        output_classes: int
            Number of output classes in the vocabulary.
        """
        super().__init__()
        self.directions = 2 if blstm else 1
        self.hidden_size = RESNET_EMBEDDING_SIZES[resnet_type]

        self.backbone = create_resnet(resnet_type, headless=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.upsample = nn.ConvTranspose2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=(1, upsampling_kern),
            stride=(1, upsampling_stride),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=blstm,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(lstm_hidden_size, output_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

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
        x = self.backbone(x)    # Batch, Channels, Height, Width // 16
        x = self.avg_pool(x)    # Batch, Channels, 1, Width // 16
        x = self.upsample(x)    # Batch, Channels, 1, Length
        x = x.squeeze(2)        # Batch, Channels, Length
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape
            x = x.view(
                seq_len,
                batch_size,
                self.directions, hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.output_layer(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x
