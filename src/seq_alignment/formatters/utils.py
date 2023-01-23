"""Miscellaneous formatting utilities."""

from typing import Any, Dict, List

import torch
from torch import nn

from base_formatter import BaseFormatter
from ..data.generic_decrypt import BatchedSample


class Compose(BaseFormatter):
    """Combine various formatters into one single call."""

    def __init__(self, formatters: List[BaseFormatter]):
        self.formatters = formatters

    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        output = []

        for fmt in self.formatters:
            current = fmt(model_output, batch)

            if not len(output):
                output = current
            else:
                output = [old | curr for old, curr in zip(output, current)]
        return output
