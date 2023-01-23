from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike

import torch
from torch import nn

from ..data.generic_decrypt import BatchedSample


class BaseMetric(ABC):
    """Compute the difference between a set of predictions and the GT."""

    @abstractmethod
    def __call__(
            self,
            output: List[Dict[str, Any]],
            batch: BatchedSample
    ) -> ArrayLike:
        """Compute the difference between a set of predictions and the GT.

        Parameters
        ----------
        model_output: List[Dict[str, Any]]
            The output of a model after being properly formatted.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        ArrayLike
            A value array that measures how far from the GT is each prediction.
        """
        raise NotImplementedError
