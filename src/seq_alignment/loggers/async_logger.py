"""Logger that performs tasks separately from the main process and thread."""

from multiprocess import Lock, Process, Pool
from pathlib import Path
from shutil import copy, move, rmtree
from typing import Any, Dict, List

import wandb

from seq_alignment.data.generic_decrypt import BatchedSample
from seq_alignment.metrics.base_metric import BaseMetric
from seq_alignment.formatters.base_formatter import BaseFormatter
from seq_alignment.loggers.base_logger import BaseLogger


class AsyncLogger(BaseLogger):
    """Asynchronous logger class that performs logging outside the main thread."""

    def __init__(
        self,
        path: Path,
        formatter: BaseFormatter,
        metric: BaseMetric,
        log_results: bool = True,
    ) -> None:
        """Set up the class and base paths.

        Parameters
        ----------
        path: Path
            The logging path for the epoch and the mode (train / valid / test) within
            the experiment.
        formatter: BaseFormatter
            Object to convert the output from a model to a processable or measurable
            output.
        metric: BaseMetric
            Object that measures fidelity to the desired processable output.
        log_results: bool = True
            Whether to store processable outputs or not (makes sense during validation
            or if this is the final output for some task, but not much during training
            aside from generating tons of superfluous data).
        """
        super().__init__(path, formatter, metric, log_results)

        self._metric_lock = Lock()
        self._result_lock = Lock()

    def process_and_log(
        self,
        output: Any,
        batch: BatchedSample,
    ) -> None:
        """Process a batch of model outputs and log them to a file.

        Parameters
        ----------
        output: Any
            The output for a single batch of the model. Must be batch-wise iterable.
        batch: BatchedSample
            An input batch with ground truth and filename data.
        """
        results = self._formatter(output, batch)
        metrics = self._metric(results, batch)

        fnames = [f.name for f in batch.filename]

        with self._metric_lock:
            self._write_metrics(metrics, fnames)
        if self._log_results:
            with self._result_lock:
                self._write_results(results, fnames)


class MultiprocessLoggerWrapper:
    """Wrapper on an AsyncLogger class that enables asynchronous parallel processing."""

    def __init__(self, logger: AsyncLogger, workers: int,) -> None:
        """Initialise wrapper.

        Parameters
        ----------
        logger: AsyncLogger
            A logging class that is apt for parallelisation.
        workers: int
            Max number of parallel processes.
        """
        self._pool = Pool(processes=workers)
        self._logger = logger

    def process_and_log(
        self,
        output: Any,
        batch: BatchedSample,
    ) -> None:
        """Asynchronously perform processing and logging of batches.

        Parameters
        ----------
        output: Any
            The output for a single batch of the model. Must be batch-wise iterable.
        batch: BatchedSample
            An input batch with ground truth and filename data.
        """
        self._pool.apply_async(self._logger.process_and_log, (output, batch))

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate logged metrics.

        Returns
        -------
        Dict[str: Any]
            A dict whose keys are aggregate names and values are the aggregations of
            values related to a metric.
        """
        self._logger.aggregate()

    def close(self):
        """Cleanup logger class and close all related files."""
        self._logger.close()
        self._pool.close()
