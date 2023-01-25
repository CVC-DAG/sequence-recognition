import json
from pathlib import Path
from typing import Any, Dict, List, Optional, List, Tuple

import tqdm
import wandb

import torch
from torch import nn
from torch.utils import data as D

from ..formatters.base_formatter import BaseFormatter
from ..metrics.base_metric import BaseMetric


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

        self.save_name = lambda epoch: f"log_{self.mode}_e{epoch}.json"

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

            fnames = []
            epoch_results = []
            epoch_metrics = []

            loss = 0.0

            for batch in tqdm(
                    self.valid_data,
                    desc=f"{self.mode} for {epoch} in Progress..."
            ):
                output = self.model.compute_batch(batch, device)
                batch_loss = self.model.compute_loss(output, batch, device)

                output = output.detach().cpu().numpy()

                results = self.formatter(output, batch)
                metrics = self.metric(results, batch)

                epoch_results += results
                epoch_metrics += metrics
                fnames += batch.filename
                loss += batch_loss

        final_metric = self.metric.aggregate(epoch_metrics)
        loss /= len(self.valid_data)
        wandb.log(
            {
                "epoch": epoch,
                "lr": self._get_lr(self.optimizer),
                f"{self.mode}_loss": batch_loss,
                self.metric.METRIC_NAME: final_metric,
            },
            step=iters,
        )

        self.log_results(fnames, epoch_results, epoch_metrics, epoch)

        try:
            first_key = list(final_metric.keys())
            first_key.sort()
            first_key = first_key[0]
            final_metric = final_metric[first_key]
        except AttributeError:
            ...

        return loss, final_metric

    def log_results(
            self,
            fnames: List[str],
            results: List[Dict[str, Any]],
            metrics: List[Dict[str, Any]],
            epoch: int,
    ) -> None:
        """Save results to a JSON file.

        Parameters
        ----------
        fnames: List[str]
            List of filenames to which each information relates to.
        results: List[Dict[str, Any]]
            Output for each model.
        metrics: List[Dict[str, Any]]
            Metrics comparing the output to the ground truth.
        epoch: int
            Epoch number.
        """
        output = {}
        for ii, (fn, rs, mt) in enumerate(zip(fnames, results, metrics)):
            output[fn] = {"results": rs, "metrics": mt}

        with open(self.save_path / self.save_name(epoch), 'w') as f_json:
            json.dump(output, f_json)
