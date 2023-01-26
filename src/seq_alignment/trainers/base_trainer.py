"""Default trainer implementation."""

import math
import json
from pathlib import Path
from typing import Any, Callable, Tuple, List, Dict, Optional, Type

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as sched
from torch.utils import data as D

from torchinfo import summary

from pydantic import BaseModel

from ..data.generic_decrypt import BatchedSample, GenericDecryptDataset
from ..formatters.base_formatter import BaseFormatter
from ..metrics.base_metric import BaseMetric
from ..models.base_model import BaseModel as BaseInferenceModel
from ..validators.base_validator import BaseValidator

from tqdm.auto import tqdm

import wandb


class BaseTrainerConfig(BaseModel):
    """Common trainer configuration."""

    batch_size: int
    device: str
    grad_clip: Optional[float]
    max_epochs: int
    learning_rate: float
    optimizer: str
    save_every: int
    eval_every: int
    weight_decay: float
    workers: int  # How many workers to use in the dataloaders

    plateau_sched: bool  # Use plateau scheduler
    plateau_factor: float  # Factor by which to reduce LR on plateau
    plateau_iters: int  # Number of epochs on which to reduce w/o improvement
    plateau_thresh: float  # Threshold to measure significant changes
    plateau_min: float  # Min LR value allowed

    warmup_sched: bool  # Use warmup scheduler
    warmup_factor: float  # Factor of reduction of lr at train start
    warmup_iters: int  # Number of iterations to increase LR at the start

    cosann_sched: bool  # Use cosine annealing scheduler
    cosann_t0: int  # Iters until first restart
    cosann_factor: int  # Factor by which to increase the number of iters until restart
    cosann_min: float  # Min learning rate


class BaseTrainer:
    """Implements default training loop and methods."""

    def __init__(
        self,
        model: BaseInferenceModel,
        train_data: D.Dataset,
        config: BaseTrainerConfig,
        save_path: Path,
        validator: BaseValidator,
        formatter: BaseFormatter,
        metric: BaseMetric,
        epoch_end_hook: Optional[Callable[List[Dict], bool]] = None,
    ) -> None:
        """Construct the BaseTrainer object with given params.

        Parameters
        ----------
        model: BaseInferenceModel
            The model with which to perform training and inference.
        train_data: D.Dataset
            Dataset to train with.
        loss_func: Callable

        config: BaseTrainerConfig

        save_path: Path

        validator: BaseValidator

        formatter: BaseFormatter

        """
        self.config = config

        self.model = model
        self.train_data = self._create_dataloader(self.config, train_data)
        self.save_path = save_path
        self.validator = validator
        self.formatter = formatter
        self.metric = metric

        self.optimizer = self._create_optimizer(self.config, self.model)
        self.warmup_sched = self._create_warmup(self.config, self.optimizer)
        self.cosann_sched = self._create_cosann(self.config, self.optimizer)
        self.plateau_sched = self._create_plateau(self.config, self.optimizer)
        self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        self.train_iters = 0

        self.epoch_end_hook = epoch_end_hook or (lambda x: True)

        self.best_metric = - math.inf if self.validator.maximise() else math.inf

        self.json_name = lambda epoch: f"log_train_e{epoch:04}.json"
        self.curr_name = lambda epoch: f"weights_e{epoch:04}.pth"
        self.best_name = "weights_BEST.pth"

    @staticmethod
    def _create_optimizer(
        config: BaseTrainerConfig,
        model: BaseInferenceModel,
    ) -> optim.Optimizer:
        """Create an optimizer based on a config object.

        Parameters
        ----------
        config: BaseTrainerConfig
            Configuration object with trainer properties.

        Returns
        -------
        optim.Optimizer:
            A torch optimizer built according to the config object.

        Raises
        ------
        ValueError
            If the queried optimizer is not available or not supported.
        """
        if config.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(
                f"The prompted optimizer ({config.optimizer}) is "
                f"not supported or does not exist."
            )
        return optimizer

    @staticmethod
    def _create_plateau(
        config: BaseTrainerConfig,
        optimizer: optim.Optimizer,
    ) -> Optional[sched.ReduceLROnPlateau]:
        """Create a Plateau Reduction scheduler based on a config object.

        Parameters
        ----------
        config: BaseTrainerConfig
            Configuration object with trainer properties.
        optimizer: optim.Optimizer
            The optimizer being employed within the trainer.

        Returns
        -------
        sched.ReduceLROnPlateau | None:
            The LR Reduction scheduler if required.
        """
        scheduler = None
        if config.plateau_sched:
            scheduler = sched.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=config.plateau_factor,
                patience=config.plateau_iters,
                threshold=config.plateau_thresh,
                min_lr=config.plateau_min,
                verbose=True,
            )

        return scheduler

    @staticmethod
    def _create_warmup(
        config: BaseTrainerConfig,
        optimizer: optim.Optimizer,
    ) -> Optional[sched.LinearLR]:
        """Create a warmup scheduler based on a config object.

        Parameters
        ----------
        config: BaseTrainerConfig
            Configuration object with trainer properties.
        optimizer: Optimizer
            The optimizer being employed within the trainer.

        Returns
        -------
        sched.LinearLR | None:
            The warmup scheduler if required.
        """
        scheduler = None
        if config.plateau_sched:
            scheduler = sched.LinearLR(
                optimizer=optimizer,
                start_factor=config.warmup_factor,
                total_iters=config.warmup_iters,
            )
        return scheduler

    @staticmethod
    def _create_cosann(
        config: BaseTrainerConfig,
        optimizer: optim.Optimizer,
    ) -> Optional[sched.CosineAnnealingWarmRestarts]:
        """Create a cosine annealing scheduler based on a config object.

        Parameters
        ----------
        config: BaseTrainerConfig
            Configuration object with trainer properties.
        optimizer: optim.Optimizer
            The optimizer being employed within the trainer.

        Returns
        -------
        sched.CosineAnnealingWarmRestarts | None:
            The cosine annealing scheduler if required.
        """
        scheduler = None
        if config.plateau_sched:
            scheduler = sched.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.cosann_t0,
                T_mult=config.cosann_factor,
                eta_min=config.cosann_min,
            )

        return scheduler

    @staticmethod
    def _create_dataloader(config: BaseTrainerConfig, dataset: D.Dataset) -> D.DataLoader:
        dataloader = D.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=bool("cuda" in config.device),
            num_workers=config.workers,
        )
        return dataloader

    @staticmethod
    def _get_lr(optimizer) -> float:
        for param_group in optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def should_training_stop(self, epoch: int, epoch_results: List[Dict]) -> bool:
        """Assess whether the training should be stopped after an epoch."""
        return False

    def save_current_weights(self, epoch: int) -> None:
        """Save current epoch model weights to trainer save directory.

        Parameters
        ----------
        epoch: int
            What epoch is currently being run.
        """
        curr_name = self.curr_name(epoch)

        self.model.save_weights(str(self.save_path / curr_name))

        prev_name = self.curr_name(epoch - self.config.save_every)
        prev_weights = self.save_path / prev_name

        if prev_weights.exists():
            prev_weights.unlink()

    def log_epoch_results(
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
            output[fn] = {
                "results": self._format_dict(rs),
                "metrics": self._format_dict(mt),
            }

        with open(self.save_path / self.json_name(epoch), 'w') as f_json:
            json.dump(output, f_json)

    def log_iteration_results(
            self
    ) -> None:
        ...

    def _format_dict(self, in_dict: Dict) -> Dict:
        for k, v in in_dict.items():
            if isinstance(v, dict):
                in_dict[k] = self._format_dict(v)
            if isinstance(v, np.ndarray):
                in_dict[k] = v.tolist()
            if isinstance(v, torch.Tensor):
                in_dict[k] = v.detach().cpu().numpy().tolist()
        return in_dict

    def train(self):
        """Perform training on the model given the trainer configuration."""
        summary(self.model)

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()

            epoch_results = []
            epoch_metrics = []
            fnames = []

            for batch in tqdm(self.train_data, desc=f"Epoch {epoch} in Progress..."):
                self.train_iters += 1

                output = self.model.compute_batch(
                    batch, self.device
                )
                batch_loss = self.model.compute_loss(
                    batch, output, self.device
                )

                self.optimizer.zero_grad()
                batch_loss.backward()

                if self.config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.optimizer.step()

                if self.warmup_sched:
                    self.warmup_sched.step()
                if self.cosann_sched:
                    self.cosann_sched.step()

                # TODO: Encapsulate this in a function and do it asynchronously

                output = output.detach().cpu().numpy()

                results = self.formatter(output, batch)
                metrics = self.metric(results, batch)

                epoch_results += results
                epoch_metrics += metrics
                fnames += batch.filename

                # Maybe not log each iteration
                wandb.log(
                    {
                        "lr": self._get_lr(self.optimizer),
                        "train_loss": batch_loss,
                    },
                    step=self.train_iters,
                )

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            self.log_epoch_results(fnames, epoch_results, epoch_metrics, epoch)

            if not epoch % self.config.save_every:
                self.save_current_weights(epoch)

            if not epoch % self.config.eval_every:
                val_loss, val_metric = self.validator.validate(
                    self.model, epoch, self.train_iters, self.device
                )

                if (self.validator.maximise() and val_metric > self.best_metric) or \
                        not (self.validator.maximise() and val_metric < self.best_metric):
                    self.best_metric = val_metric
                    self.best_epoch = epoch

                    self.model.save_weights(str(self.save_path / self.best_name))

            self.epoch_end_hook(epoch_results)
            if self.should_training_stop(epoch, epoch_results):
                break
