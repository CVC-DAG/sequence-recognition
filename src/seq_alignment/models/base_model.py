"""Base model class with QoL functions already implemented."""

from typing import Optional
from warnings import warn

from pydantic import BaseModel as ConfigBaseModel
import torch
from torch import nn

from ..data.generic_decrypt import BatchedSample


class BaseModelConfig(ConfigBaseModel):
    """Base configuration for a model."""

    model_name: str
    model_weights: Optional[str]


class BaseModel(nn.Module):
    """Template for a model with some QoL methods available.

    A template to follow by models used in inference on the Decrypt
    alignment codebase. The model must own the loss function as well.
    """

    MODEL_CONFIG = BaseModelConfig

    def __init__(self) -> None:
        """Initialise Model."""
        super().__init__()

    def load_weights(self, wpath: str) -> None:
        """Load a set of weights into the model.

        Parameters
        ----------
        wpath: str
            Path to the set of weights to load into the model.
        """
        weights = torch.load(wpath)
        missing, unexpected = self.load_state_dict(weights, strict=False)

        if missing or unexpected:
            warn("Careful: Not all weights have been loaded on the model")
            print("Missing: ", missing)
            print("Unexpected: ", unexpected)

    def save_weights(self, path: str) -> None:
        """Save the weights of the model unto the given path.

        Parameters
        ----------
        path: str
            Path to store the model's weights.
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def compute_batch(self, batch: BatchedSample, device: torch.device) -> torch.Tensor:
        """Generate the model's output for a single input batch.

        Parameters
        ----------
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        device: torch.device
            Device where the training is happening in order to move tensors.

        Returns
        -------
        output: torch.Tensor
            The output of the model for the input batch.
        """
        raise NotImplementedError

    def compute_loss(
        self, batch: BatchedSample, output: torch.Tensor, device: torch.device
    ) -> torch.float32:
        """Generate the model's loss for a single input batch and output.

        Parameters
        ----------
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        output: torch.Tensor
            The output of the model for the input batch.
        device: torch.device
            Device where the training is happening in order to move tensors.

        Returns
        -------
        torch.float32
            The model's loss for the given input.
        """
        raise NotImplementedError
