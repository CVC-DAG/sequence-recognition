"""CRNN-based models."""
from __future__ import annotations

import torch
from torch import nn
from torch.typing import TensorType
from warnings import warn

from cnns import create_resnet


class RecurrentCTC(nn.Module):
    def __init__(
            self,
            resnet_type: int,
            lstm_layers: int,
            lstm_input_size: int,
            lstm_hidden_size: int,
            blstm: bool,
            dropout: float,
            output_classes: int,
    ) -> None:
        self.backbone = create_resnet(resnet_type, headless=True)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=blstm,
            dropout=dropout
        )
        self.output_layer = nn.Linear(
            lstm_hidden_size,
            output_classes
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
            self,
            x: TensorType
    ) -> TensorType:
        raise NotImplementedError

    def load_weights(self, wpath: str) -> None:
        weights = torch.load(wpath)
        missing, unexpected = self.load_state_dict(weights, strict=False)

        if missing or unexpected:
            warn("Careful: Not all weights have been loaded on the model")
            print("Missing: ", missing)
            print("Unexpected: ", unexpected)

    def save_weights(self, path: str) -> None:
        state_dict = self.state_dict()
        torch.save(
            state_dict,
            path
        )
