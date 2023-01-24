"""Miscellaneous metric utilities."""

from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike

from .base_metric import BaseMetric
from ..data.generic_decrypt import BatchedSample


class Compose(BaseMetric):
    """Combine various metrics into one single call."""

    METRIC_NAME = "compose"

    def __init__(self, metrics: List[BaseMetric]):
        """Initialise composition of metrics.

        Parameters
        ----------
        metrics: List[BaseMetric]
            List of metrics to be computed for a single output.
        """
        self.metrics = metrics

    def __call__(
            self,
            output: List[Dict[str, Any]],
            batch: BatchedSample
    ) -> Dict[str, ArrayLike]:
        """Compute multiple metrics between a set of predictions and the GT.

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
        output = []

        for metric in self.metrics:
            current = metric(output, batch)

            if not len(output):
                output = current
            else:
                output = [old | curr for old, curr in zip(output, current)]
        return output

    def aggregate(self, metrics: Dict[str, ArrayLike]) -> float:
        """Aggregate a set of predictions to return the average edit distance.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        Dict[str, float]
            Average of seqiou predictions for all bounding boxes.
        """
        output = {}

        for metric in self.metrics:
            output[metric.METRIC_NAME] = metric.aggregate(metrics)

        return output
