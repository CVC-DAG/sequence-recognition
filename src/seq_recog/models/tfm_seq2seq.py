"""Transformer-based Encoder-Decoder models."""

from typing import List, Optional

import torch
from torch import nn
from torch import TensorType

from .losses.focal_loss import SequenceFocalLoss
from .base_model import BaseModel
from .base_model import BaseModelConfig
from ..data.base_dataset import BatchedSample, BaseDataConfig, BaseVocab


class TransformerSeq2SeqConfig(BaseModelConfig):
    """Settings for Transformer-Based Seq2Seq models."""

    loss_function: str = "cross-entropy"
    focal_loss_gamma: float = 1.0
    label_smoothing: float = 0.0
    loss_weights: Optional[List[float]] = None


class TransformerSeq2Seq(BaseModel):
    """Transformer-based Sequence to Sequence transcription models."""

    MODEL_CONFIG = TransformerSeq2SeqConfig

    def __init__(
        self, model_config: TransformerSeq2SeqConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Model."""
        super().__init__(model_config, data_config)

        self.tgt_mask = nn.Parameter(
            self._get_tgt_mask(data_config.target_seqlen),
            requires_grad=False,
        )

        if model_config.loss == "focal":
            self.loss = SequenceFocalLoss(
                gamma=model_config.focal_loss_gamma,
                label_smoothing=model_config.label_smoothing,
                ignore_index=BaseVocab.PAD_INDEX,
                class_weights=torch.tensor(model_config.loss_weights),
            )
        else:
            self.loss = nn.CrossEntropyLoss(
                weight=torch.tensor(model_config.loss_weights),
                ignore_index=BaseVocab.PAD_INDEX,
                label_smoothing=model_config.label_smoothing,
            )

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

    @staticmethod
    def _get_tgt_mask(seqlen: int) -> TensorType:
        mask = torch.ones(seqlen - 1, seqlen - 1)
        mask = (torch.triu(mask) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask
