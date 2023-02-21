import json
from pathlib import Path
from typing import Any, Dict, List, Optional, List, Tuple, Type
import numpy as np

from tqdm.auto import tqdm
import wandb

import torch
from torch import nn
from torch.utils import data as D

from ..formatters.base_formatter import BaseFormatter
from ..metrics.base_metric import BaseMetric
from ..loggers.base_logger import BaseLogger
from ..loggers.async_logger import AsyncLogger


class BaseValidator:
    """Validator class for an experiment."""

    def __init__(
            self,
            valid_data: D.Dataset,
            valid_formatter: BaseFormatter,
            valid_metric: BaseMetric,
            save_path: Path,
            batch_size: int,
            workers: int,
            mode: str,
            logger: Type[BaseLogger] = AsyncLogger,
    ) -> None:
        """Construct validator.

        Parameters
        ----------
        valid_data: D.Dataset
            Dataset to validate with.
        valid_formatter: BaseFormatter
            Formatter to convert the output from the model.
        valid_metric: BaseMetric
            Metric to evaluate the results from validation.
        save_path: Path
            Output to save logging stuff.
        batch_size: int
            Batch size to perform validation with.
        workers: int
            Number of workers for the dataloader.
        mode: str
            Validation or test.
        logger: Type[BaseLogger] = AsyncLogger
            A logging class to save results or measure stuff.
        """
        self.valid_data = D.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=workers,
        )
        self.valid_formatter = valid_formatter
        self.valid_metric = valid_metric
        self.save_path = save_path
        self.mode = mode
        self.workers = workers

        self.logger_type = logger

    def maximise(self) -> bool:
        """Return whether the underlying metric is maximising."""
        return self.valid_metric.maximise()

    def validate(
            self,
            model: nn.Module,
            epoch: int,
            iters: int,
            device: torch.device,
    ) -> Tuple[float, float]:
        """Perform validation on an instance of the trained model.

        Parameters
        ----------
        model: nn.Module
            Model to evaluate.
        epoch: int
            Number of the epoch on the model instance.
        iters: int
            Training iteration of the model instance.

        Returns
        -------
        float
            Loss value for validation.
        float
            Aggregate metric for validation.
        """
        with torch.no_grad():
            model.eval()

            log_path = self.save_path / f"e{epoch}_valid"

            logger = self.logger_type(
                log_path,
                self.valid_formatter,
                self.valid_metric,
                True,
                self.workers,
            )
            log_path.mkdir(exist_ok=True)

            loss = 0.0

            for batch in tqdm(
                    self.valid_data,
                    desc=f"{self.mode} for {epoch} in Progress..."
            ):
                output = model.compute_batch(batch, device)
                batch_loss = model.compute_loss(batch, output, device)

                output = output.detach().cpu().numpy()
                logger.process_and_log(output, batch)

        logger.close()
        agg_metrics = logger.aggregate()
        final_metric = agg_metrics[next(iter(agg_metrics), None)]

        loss /= len(self.valid_data)
        wandb.log(
            {
                "epoch": epoch,
                f"{self.mode}_loss": batch_loss,
            } | {f"{self.mode}_{k}": v for k, v in agg_metrics.items()},
            step=iters,
        )

        with open(log_path / "summary.json", 'w') as f_summary:
            json.dump(agg_metrics, f_summary, indent=4)

        return loss, final_metric
