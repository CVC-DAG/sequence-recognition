"""Base model class with QoL functions already implemented."""

from typing import Dict
from warnings import warn

import torch
from torch import nn
from ..data.generic_decrypt import BatchedSample


class BaseModel(nn.Module):
    """Template for a Decrypt model.

    A template to follow by models used in inference on the Decrypt
    alignment codebase.
    """

    def __init__(self) -> None:
        """Initialise BaseModel."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference on the model."""
        raise NotImplementedError

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

    def compute_batch(self, batch: BatchedSample) -> torch.Tensor:
        """Generate the model's output for a single input batch.

        Parameters
        ----------
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.

        Returns
        -------
        output: torch.Tensor
            The output of the model for the input batch.
        """
        raise NotImplementedError

    def compute_results(
        self, model_output: torch.Tensor, batch: BatchedSample
    ) -> Dict:
        """Compute the human-readable version of the model's batch output.

        Parameters
        ----------
        output: torch.Tensor
            The output for a single training iteration for the model
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.

        Returns
        -------
        dict
            The output of the model for the input batch. The key-value pairs
            depend on the underlying model.
        """
        raise NotImplementedError
