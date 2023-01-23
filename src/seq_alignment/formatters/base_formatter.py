"""Base formatter class implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import nn

from ..data.generic_decrypt import BatchedSample


class BaseFormatter(ABC):
    """Abstracts converting from a model output to the desired format."""

    @abstractmethod
    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Convert a model output to any other formatting.

        Parameters
        ----------
        model_output: torch.Tensor
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dicts where keys are the names of the formatting
            techniques and the values are the formatted outputs.
        """
        raise NotImplementedError
