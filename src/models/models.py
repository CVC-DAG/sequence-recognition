from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Tuple

from torch import nn
import torch

from ..data.generic_decrypt import GenericSample


class ModelOutput(NamedTuple):
    output_coords: torch.Tensor
    output_seqs: torch.Tensor
    loss: torch.Tensor


class GenericModel(ABC):
    def __init__(
            self,
            model: nn.Module,
    ):
        self._model = model

    @abstractmethod
    def inference(
            self,
            batch: GenericSample
    ) -> ModelOutput:
        raise NotImplementedError

