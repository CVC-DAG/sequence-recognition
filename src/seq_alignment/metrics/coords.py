"""Coordinate-based metrics."""

from .base_metric import BaseMetric
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
from ..data.generic_decrypt import BatchedSample
from ..utils.ops import seqiou


class SeqIoU(BaseMetric):
    """Sequence-level Intersection over Union metric."""

    METRIC_NAME = "sequence_iou"

    def __call__(
            self,
            output: List[Dict[str, Any]],
            batch: BatchedSample
    ) -> Dict[str, ArrayLike]:
        """Compute the IoU of the output sequences and the ground truth.

        Parameters
        ----------
        model_output: List[Dict[str, Any]]
            The output of a model after being properly formatted.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            The IoU for each bounding box for each element in the sequence.
        """
        out = []

        for out, gt in zip(output, batch):
            iou = seqiou(out["prediction"].get_coords(), gt.segm)
            out.append({"seqiou": iou})

        return out

    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        return True

    def aggregate(self, metrics: Dict[str, ArrayLike]) -> float:
        """Aggregate a set of predictions to return the average seqiou.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        float
            Average of seqiou predictions for all bounding boxes.
        """
        preds = np.concatenate([pred["seqiou"] for pred in metrics])
        return np.mean(preds)
