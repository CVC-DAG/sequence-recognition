"""Base experiment types and configs."""

from typing import Dict
from pathlib import Path
from pydantic import BaseModel

from ..data.generic_decrypt import DataConfig
from ..models.base_model import BaseModelConfig
from ..trainers.base_trainer import BaseTrainerConfig


class DirectoryConfig(BaseModel):
    """Directory settings."""

    results_dir: str
    base_data_dir: str

    training_file: str
    training_root: str
    validation_file: str
    validation_root: str
    test_file: str
    test_root: str

    vocab_data: str


class ExperimentConfig(BaseModel):
    """Global experiment settings."""

    exp_name: str
    description: str
    cipher: str
    wandb_mode: str
    wandb_project: str

    dirs: BaseModel
    data: DataConfig
    model: BaseModelConfig
    train: BaseTrainerConfig


class BaseExperiment:
    def __init__() -> None:
        ...


