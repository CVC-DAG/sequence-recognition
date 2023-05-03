"""Experiment with Arnau Baró's CRNN model."""
import torch

from pathlib import Path
from seq_alignment.data.generic_decrypt import (
    GenericDecryptVocab,
    GenericDecryptDataset,
    DataConfig,
)
from seq_alignment.experiments.base_experiment import Experiment, ExperimentConfig
from seq_alignment.experiments.configurations import DecryptDirectoryConfig
from seq_alignment.formatters import ctc_formatters
from seq_alignment.loggers.base_logger import SimpleLogger
from seq_alignment.loggers.async_logger import AsyncLogger
from seq_alignment.metrics import coords, text
from seq_alignment.models.ctc_models import BaroCRNN, BaroCRNNConfig
from seq_alignment.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from seq_alignment.validators.base_validator import BaseValidator


class BaroExperimentConfig(ExperimentConfig):
    """Global experiment settings."""

    beam_width: int

    dirs: DecryptDirectoryConfig
    data: DataConfig
    model: BaroCRNNConfig
    train: BaseTrainerConfig


class BaroExperiment(Experiment):
    """Object modelling the Experiment with Arnau Baró's CRNN model."""

    EXPERIMENT_CONFIG = BaroExperimentConfig

    def __init__(self):
        """Initialise object."""
        super().__init__()

    def initialise_everything(self) -> None:
        """Initialise all member variables for the class."""
        # Data
        self.vocab = GenericDecryptVocab(self.cfg.dirs.vocab_data)
        self.train_data = GenericDecryptDataset(
            self.cfg.dirs.training_root,
            self.cfg.dirs.training_file,
            self.vocab,
            self.cfg.data,
            True,
        )
        self.valid_data = GenericDecryptDataset(
            self.cfg.dirs.validation_root,
            self.cfg.dirs.validation_file,
            self.vocab,
            self.cfg.data,
            False,
        )
        self.test_data = GenericDecryptDataset(
            self.cfg.dirs.test_root,
            self.cfg.dirs.test_file,
            self.vocab,
            self.cfg.data,
            False,
        )

        # Formatters
        self.training_formatter = ctc_formatters.GreedyTextDecoder()
        self.valid_formatter = ctc_formatters.GreedyTextDecoder()

        # Metrics
        self.training_metric = text.Levenshtein(self.vocab)
        self.valid_metric = text.Levenshtein(self.vocab)

        # Model and training-related
        self.model = BaroCRNN(self.cfg.model, self.cfg.data)
        self.validator = BaseValidator(
            self.valid_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "valid",
            SimpleLogger,
        )
        self.tester = BaseValidator(
            self.test_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "test",
            SimpleLogger,
        )

        self.trainer = BaseTrainer(
            self.model,
            self.train_data,
            self.cfg.train,
            Path(self.cfg.dirs.results_dir),
            self.validator,
            self.training_formatter,
            self.training_metric,
            None,
            SimpleLogger,
        )


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    exp = BaroExperiment()
    exp.main()
