"""Implementation of a base metric object."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from numpy.typing import ArrayLike

from ..data.generic_decrypt import BatchedSample


class BaseMetric(ABC):
    """Compute the difference between a set of predictions and the GT."""

    METRIC_NAME = "Base Metric"

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, metrics: Dict[str, ArrayLike]) -> float:
        """Aggregate a set of predictions from the given metric.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        float
            An aggregate value summarising the entire prediction.
        """
        raise NotImplementedError
