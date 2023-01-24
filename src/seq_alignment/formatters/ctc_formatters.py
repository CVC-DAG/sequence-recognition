"""Implementation of conversions from a CTC output model to anything else."""

from itertools import groupby
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import nn

from base_formatter import BaseFormatter
from ..data.generic_decrypt import BatchedSample
from ..utils.decoding import PrefixTree, Prediction


class OptimalCoordinateDecoder(BaseFormatter):
    """From a CTC matrix, get the optimal decoding sequence using the GT."""

    def __init__(self, beam_width: int) -> None:
        """Construct OptimalCoordinateDecoding object.

        Parameters
        ----------
        beam_width: int
            Number of beams to expand in order to find the optimal decoding.
        """
        super().__init__()
        self.beam_width = beam_width

    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Convert a model output to a sequence of coordinate predictions.

        Parameters
        ----------
        model_output: torch.Tensor
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Prediction]]
            A List of Prediction objects with character coordinates.
        """
        outputs = []
        model_output = model_output.transpose((1, 0, 2))
        batch_size, columns, classes = model_output.shape
        target_shape = batch.img.shape[-1]
        widths = batch.curr_shape[0]

        csizes = [ww * (columns / target_shape) for ww in widths]

        for mat, transcript, csize in zip(model_output, batch.gt, csizes):
            tree = PrefixTree(
                transcript,
                self.beam_width
            )
            decoding = tree.decode(mat)
            prediction = Prediction.from_ctc_decoding(
                decoding,
                transcript,
                mat,
                csize,
            )
            outputs.append({"prediction": prediction})
        return outputs


class GreedyTextDecoder(BaseFormatter):
    """Generate an unpadded token sequence from a CTC output."""

    def __init__(self) -> None:
        """Construct GreedyTextDecoder object."""
        super().__init__()

    def __call__(
            self,
            model_output: torch.Tensor,
            batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Convert a model output to a token sequence.

        Parameters
        ----------
        model_output: torch.Tensor
            The output of a model.
        batch: BatchedSample
            Batch information.

        Returns
        -------
        List[Dict[str, ArrayLike]]
            A List of sequences of tokens corresponding to the decoded output.
        """
        model_output = model_output.transpose((1, 0, 2))
        output = []
        for sample in model_output:
            maximum = sample.argmax(axis=-1)
            output.append(
                {"text": np.array([k for k, g in groupby(maximum) if k != 0])})
        return output
