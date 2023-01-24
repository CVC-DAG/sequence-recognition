"""Text-based metrics."""

from .base_metric import BaseMetric
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
from ..data.generic_decrypt import BatchedSample, GenericDecryptVocab
from ..utils.ops import levenshtein


class Levenshtein(BaseMetric):
    """Levenshtein metric."""

    METRIC_NAME = "levenshtein"

    def __init__(self, vocab: GenericDecryptVocab) -> None:
        self.vocab = vocab

    def __call__(
            self,
            output: List[Dict[str, Any]],
            batch: BatchedSample
    ) -> Dict[str, ArrayLike]:
        """Compute the difference between a set of predictions and the GT.

        Parameters
        ----------
        model_output: List[Dict[str, Any]]
            The output of a model after being properly formatted.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            A value array that measures how far from the GT is each prediction.
        """
        out = []

        for out, gt in zip(output, batch):
            lev = levenshtein(out["text"], self.vocab.unpad(gt.gt))
            out.append({"levenshtein": lev})

        return out

    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        return False

    def aggregate(self, metrics: Dict[str, ArrayLike]) -> float:
        """Aggregate a set of predictions to return the average edit distance.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        float
            Average of seqiou predictions for all bounding boxes.
        """
        preds = np.array([pred["levenshtein"] for pred in metrics])
        return np.mean(preds)