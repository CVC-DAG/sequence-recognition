"""Miscellaneous formatting utilities."""

from typing import Any, Dict, List

import torch
from torch import nn

from .base_formatter import BaseFormatter
from ..data.generic_decrypt import BatchedSample


class Compose(BaseFormatter):
    """Combine various formatters into one single call."""

    def __init__(self, formatters: List[BaseFormatter]):
        """Initialise composition of formatters.

        Parameters
        ----------
        formatters: List[BaseFormatter]
            List of formatters to be computed for a single output.
        """
        self.formatters = formatters

    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Compute multiple formatters between a set of predictions and the GT.

        Parameters
        ----------
        model_output: torch.Tensor
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Any]]
            List of formatted predictions.
        """
        output = []

        for fmt in self.formatters:
            current = fmt(model_output, batch)

            if not len(output):
                output = current
            else:
                output = [old | curr for old, curr in zip(output, current)]
        return output


class AddFilename(BaseFormatter):
    """Add filename into the formatting dictionary."""

    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Provide the filename to the formatted dict.

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
        return [x for x in batch.filename]
