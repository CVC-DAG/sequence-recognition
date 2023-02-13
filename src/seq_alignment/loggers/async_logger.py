"""Logger that performs tasks separately from the main process and thread."""

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
        workers: int,
        log_results: bool = True,
    ) -> None:
        """Initialise wrapper.

        Parameters
        ----------
        logger: AsyncLogger
            A logging class that is apt for parallelisation.
        workers: int
            Max number of parallel processes. Note that two extra processes are created
            for writers.
        """
        self._mgr = mp.Manager()
        self._pool = mp.Pool(processes=workers + 2)

        self._metric_queue = self._mgr.Queue()
        self._result_queue = self._mgr.Queue()

        self._formatter = formatter
        self._metric = metric
        self._log_results = log_results

        self._processor = AsyncProcessor(
            self._formatter, self._metric, self._log_results
        )

        _ = self._pool.apply_async(
            __writer, (path, self._metric.keys(), "metric", self._metric_queue)
        )
        if self._log_results:
            _ = self._pool.apply_async(
                __writer, (path, self._formatter.keys(), "results", self._result_queue)
            )

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
        self._pool.apply_async(
            self._processor, (output, batch, self._metric_queue, self._result_queue)
        )

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate logged metrics.

        Returns
        -------
        Dict[str: Any]
            A dict whose keys are aggregate names and values are the aggregations of
            values related to a metric.
        """
        return {}

    def close(self):
        """Cleanup logger class and close all related files."""
        self._metric_queue.put("kill")
        self._result_queue.put("kill")

        self._logger.close()
        self._pool.close()


def __writer(
    path: Path,
    names: List[str],
    fname_base: str,
    q: mp.Queue,
) -> None:
    metric_paths = {
        name: open(path / f"{fname_base}_{name}.npz", "rw") for name in names
    }

    while True:
        content = q.get()
        if content == "kill":
            for v in metric_paths.values():
                v.close()
            return
        img_names, output = content
        for fn, out in zip(img_names, output):
            for k, v in out.items():
                np.savez_compressed(metric_paths[k], fn=v)


class AsyncProcessor:
    """Performs output conversion and metric computations."""

    def __init__(
        self,
        formatter: BaseFormatter,
        metric: BaseMetric,
        log_output: bool,
    ) -> None:
        """Construct async processor object.

        Parameters
        ----------
        formatter: BaseFormatter
            Result formatting object.
        metric: BaseMetric
            Metric computation object.
        log_output: bool
            Whether or not to log formatted outputs.
        """
        self._formatter = formatter
        self._metric = metric
        self._log_results = log_output

    def __call__(
        self,
        output: Any,
        batch: BatchedSample,
        metric_q: mp.Queue,
        result_q: mp.Queue,
    ) -> None:
        """Call the processor in order to generate results and write them to output pipes.

        Parameters
        ----------
        output: Any
            Model output.
        batch: BatchedSample
            Batch of input data.
        metric_q: mp.Queue
            Queue to write metric results.
        result_q: mp.Queue
            Queue to write formatted results.
        """
        fnames = batch.filename
        results = self._formatter(output, batch)
        if self._log_results:
            result_q.put((fnames, results))
        metrics = self._metric(output, batch)
        metric_q.put((fnames, metrics))
