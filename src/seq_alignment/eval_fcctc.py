"""Evaluate Fully Convolutional CTC model."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

import torch
from torch import nn

from data.generic_decrypt import (
    GenericDecryptDataset,
    GenericDecryptVocab,
)
from models.cnns import FullyConvCTC
from utils.decoding import decode_ctc
from align_fcctc import Config


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        data: 
    ) -> None:
        self._model = model

    def eval(self) -> None:
        
