"""Default trainer implementation."""

from pathlib import Path
from typing import Callable, Tuple, List, Dict, Optional

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as sched
from torch.utils import data as D

from torchinfo import summary

from pydantic import BaseModel

from data.generic_decrypt import BatchedSample, GenericDecryptDataset
from models.base_model import BaseModel
from base_validator import Validator

from tqdm.auto import tqdm

import wandb


class TrainerConfig(BaseModel):
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


class Trainer:
    """Implements default training loop and methods."""

    def __init__(
        self,
        model: BaseModel,
        train_data: D.Dataset,
        loss_func: Callable,
        config: TrainerConfig,
        save_path: Path,
        validator: Validator,
        epoch_end_hook: Callable[List[Dict], bool] | None = None,
    ) -> None:
        self.config = config

        self.model = model
        self.train_data = self._create_dataloader(self.config, train_data)
        self.loss_func = loss_func
        self.save_path = save_path

        self.optimizer = self._create_optimizer(self.config, self.model)
        self.warmup_sched = self._create_warmup(self.config, self.optimizer)
        self.cosann_sched = self._create_cosann(self.config, self.optimizer)
        self.plateau_sched = self._create_plateau(self.config, self.optimizer)
        self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        self.train_iters = 0

        self.epoch_end_hook = epoch_end_hook or (lambda x: True)
        self.curr_name = lambda epoch: f"weights_e{epoch}.pth"

    @staticmethod
    def _create_optimizer(
        config: TrainerConfig,
        model: BaseModel,
    ) -> optim.Optimizer:
        """Create an optimizer based on a config object.

        Parameters
        ----------
        config: TrainerConfig
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
            raise ValueError(f"The prompted optimizer ({config.optimizer}) is "
                             f"not supported or does not exist.")
        return optimizer

    @staticmethod
    def _create_plateau(
        config: TrainerConfig,
        optimizer: optim.Optimizer,
    ) -> sched.ReduceLROnPlateau | None:
        """Create a Plateau Reduction scheduler based on a config object.

        Parameters
        ----------
        config: TrainerConfig
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
        config: TrainerConfig,
        optimizer: optim.Optimizer,
    ) -> sched.LinearLR | None:
        """Create a warmup scheduler based on a config object.

        Parameters
        ----------
        config: TrainerConfig
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
        config: TrainerConfig,
        optimizer: optim.Optimizer,
    ) -> sched.CosineAnnealingWarmRestarts | None:
        """Create a cosine annealing scheduler based on a config object.

        Parameters
        ----------
        config: TrainerConfig
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
    def _create_dataloader(
            config: TrainerConfig, dataset: D.Dataset
    ) -> D.DataLoader:
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

    def compute_loss(
        self, model_output: torch.Tensor, batch_in: BatchedSample
    ) -> torch.float32:
        """Compute the model's loss for a single batch output.

        Parameters
        ----------
        output: torch.Tensor
            The output for a single training iteration for the model
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.

        Returns
        -------
        output: torch.Tensor
            The output of the model for the input batch.
        """
        raise NotImplementedError

    def should_training_stop(self, epoch_results: List[Dict]) -> bool:
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

    def train(self):
        """Perform training on the model given the trainer configuration."""
        summary(self.model)

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()

            epoch_results = []

            for batch in tqdm(
                    self.train_data,
                    desc=f"Epoch {epoch} in Progress..."
            ):
                self.train_iters += 1

                output = self.model.compute_batch(batch)
                batch_loss = self.compute_loss(output, batch)

                results.append(epoch_results)

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

                wandb.log(
                    {
                        "lr": self._get_lr(self.optimizer),
                        "train_loss": batch_loss,
                    },
                    step=self.train_iters,
                )

            if not epoch % self.config.save_every:
                self.save_current_weights(epoch)

            if not epoch % self.config.eval_every:
                val_loss, val_iou = self.validator(epoch)

                if val_iou > self.best_iou:
                    self.best_iou = val_iou
                    self.best_epoch = epoch

                    self.model.save_weights(str(self.save_path / self.best_name))

            self.epoch_end_hook(epoch_results)
            if self.should_training_stop(epoch_results):
                break
