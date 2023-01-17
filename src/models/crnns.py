"""CRNN-based models."""
from __future__ import annotations

import torch
from torch import nn
from warnings import warn

from .cnns import create_resnet, RESNET_EMBEDDING_SIZES, BaroCNN


class BaroCRNN(nn.Module):
    def __init__(
        self,
        lstm_hidden_size: int,
        lstm_layers: int,
        blstm: bool,
        dropout: float,
        output_classes: int,
    ) -> None:
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
        x = self.backbone(x)
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape

            x = x.view(seq_len, batch_size, self.directions, hidden_size // self.directions)
            x = x.sum(axis=2)

        x = self.linear(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x

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


class ResnetCRNN(nn.Module):
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
        x = self.backbone(x)    # Batch, Channels, Height, Width // 16
        x = self.avg_pool(x)    # Batch, Channels, 1, Width // 16
        x = self.upsample(x)    # Batch, Channels, 1, Length
        x = x.squeeze(2)        # Batch, Channels, Length
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape
            x = x.view(seq_len, batch_size, self.directions, hidden_size // self.directions)
            x = x.sum(axis=2)

        x = self.linear(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x

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
