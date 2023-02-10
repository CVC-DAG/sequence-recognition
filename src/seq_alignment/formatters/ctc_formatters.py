"""Implementation of conversions from a CTC output model to anything else."""

from itertools import groupby
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import nn

from .base_formatter import BaseFormatter
from ..data.generic_decrypt import BatchedSample, GenericDecryptVocab
from ..utils.decoding import PrefixTree, Prediction


class OptimalCoordinateDecoder(BaseFormatter):
    """From a CTC matrix, get the optimal decoding sequence using the GT."""

    KEY_COORD1D = "coords1d"
    KEY_COORD1D_CONF = "coords1d_confidences"
    KEYS = [KEY_COORD1D, KEY_COORD1D_CONF]

    def __init__(self, beam_width: int, vocab: GenericDecryptVocab) -> None:
        """Construct OptimalCoordinateDecoding object.

        Parameters
        ----------
        beam_width: int
            Number of beams to expand in order to find the optimal decoding.
        """
        super().__init__()
        self.beam_width = beam_width
        self.vocab = vocab

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
        curr_shape = batch.curr_shape[0].numpy()
        og_shape = batch.og_shape[0].numpy()

        csizes = (target_shape / columns) * (og_shape / curr_shape)

        for mat, transcript, csize in zip(model_output, batch.gt, csizes):
            transcript = np.array(self.vocab.unpad(transcript.numpy()))
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
            outputs.append({self.KEY_COORD1D: prediction.get_coords(),
                            self.KEY_COORD1D_CONF: prediction.get_confidences()})
        return outputs


class GreedyTextDecoder(BaseFormatter):
    """Generate an unpadded token sequence from a CTC output."""

    KEY_TEXT = "coords1d"
    KEY_TEXT_CONF = "coords1d_confidences"
    KEYS = [KEY_TEXT, KEY_TEXT_CONF]

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
            The output of a CTC model. Should be a L x B x C matrix, where L is the
            sequence length, B is the batch size and C is the number of classes.
        batch: BatchedSample
            Batch information.

        Returns
        -------
        List[Dict[str, ArrayLike]]
            A List of sequences of tokens corresponding to the decoded output and the
            output confidences encapsulated within a dictionary.
        """
        model_output = model_output.transpose((1, 0, 2))
        indices = model_output.argmax(axis=-1)
        output = []

        for sample, mat in zip(indices, model_output):
            nonzero = sample != 0
            text_ind = sample[nonzero]
            text_cnf = mat[nonzero]
            output.append({self.KEY_TEXT: text_ind,
                           self.KEY_TEXT_CONF: text_cnf})
        return output
