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


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError
