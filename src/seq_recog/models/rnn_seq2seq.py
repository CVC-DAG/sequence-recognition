"""RNN-based Encoder-Decoder models."""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import TensorType
from torch.autograd import Variable
from torch.nn.functional import one_hot

from .losses.focal_loss import SequenceFocalLoss
from .base_model import BaseModel
from .base_model import BaseModelConfig
from ..data.base_dataset import BatchedSample, BaseDataConfig, BaseVocab

from . import rnn_encoders as rnne
from . import rnn_decoders as rnnd
from . import rnn_attention as rnna


class RNNSeq2SeqConfig(BaseModelConfig):
    """Settings for Transformer-Based Seq2Seq models."""

    loss_function: str = "cross-entropy"
    focal_loss_gamma: float = 1.0
    focal_loss_alpha: float = 1.0
    label_smoothing: float = 0.0
    loss_weights: Optional[List[float]] = None


class RNNSeq2Seq(BaseModel):
    """Transformer-based Sequence to Sequence transcription models."""

    MODEL_CONFIG = RNNSeq2SeqConfig

    def __init__(
        self, model_config: RNNSeq2SeqConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Model."""
        super().__init__(model_config, data_config)

        if model_config.loss_function == "focal":
            self.loss = SequenceFocalLoss(
                gamma=model_config.focal_loss_gamma,
                alpha=model_config.focal_loss_alpha,
                label_smoothing=model_config.label_smoothing,
                ignore_index=BaseVocab.PAD_INDEX,
                class_weights=torch.tensor(model_config.loss_weights)
                if model_config.loss_weights is not None
                else None,
            )
        else:
            self.loss = nn.CrossEntropyLoss(
                weight=torch.tensor(model_config.loss_weights)
                if model_config.loss_weights is not None
                else None,
                ignore_index=BaseVocab.PAD_INDEX,
                label_smoothing=model_config.label_smoothing,
            )

    def compute_batch(self, batch: BatchedSample, device: torch.device) -> torch.Tensor:
        """Generate the model's output for a single input batch.

        Parameters
        ----------
        batch: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        device: torch.device
            Device where the training is happening in order to move tensors.

        Returns
        -------
        output: torch.Tensor
            The output of the model for the input batch.
        """
        images = batch.img.to(device)
        transcript = batch.gt.to(device)
        lengths = batch.curr_shape[0].to(device)

        output, _ = self.forward(images, transcript, lengths)
        return output

    def compute_loss(
        self, batch: BatchedSample, output: torch.Tensor, device: torch.device
    ) -> torch.float32:
        """Generate the model's loss for a single input batch and output.

        Parameters
        ----------
        batch: BatchedSample
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
        output = output.reshape(-1, output.shape[-1])
        transcript = batch.gt.to(device)[:, 1:].reshape(-1)

        return self.loss(output, transcript)


class KangSeq2SeqConfig(RNNSeq2SeqConfig):
    """Configuration of Lei Kang-inspired RNN Sequence to Sequence model."""

    vgg_type: int
    vgg_bn: bool
    vgg_pretrain: bool
    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    attn_filters: int
    attn_kernsize: int
    attn_smoothing: bool = False
    embedding_size: int
    output_units: int
    teacher_rate: float
    dropout: float
    tradeoff_context_embedding: Optional[float] = None
    multinomial: bool = False


class KangSeq2Seq(RNNSeq2Seq):
    """Lei Kang [1]-inspired Sequence to Sequence model.

    [1] L. Kang, J. I. Toledo, P. Riba, M. Villegas, A. Fornés, and M. Rusiñol,
    "Convolve, Attend and Spell: An Attention-based Sequence-to-Sequence Model for
    Handwritten Word Recognition," in Pattern Recognition, T. Brox, A. Bruhn, and
    M. Fritz, Eds., in Lecture Notes in Computer Science. Cham: Springer International
    Publishing, 2019, pp. 459-472. doi: 10.1007/978-3-030-12939-2_32.
    """

    def __init__(
        self,
        model_config: RNNSeq2SeqConfig,
        data_config: BaseDataConfig,
    ) -> None:
        super().__init__(model_config, data_config)

        self.encoder = rnne.VggRNNEncoder(
            vgg_type=model_config.vgg_type,
            vgg_bn=model_config.vgg_bn,
            vgg_pretrain=model_config.vgg_pretrain,
            hidden_size=model_config.hidden_size,
            layers=model_config.encoder_layers,
            height=data_config.target_shape[1],
            width=data_config.target_shape[0],
            dropout=model_config.dropout,
        )
        attention = rnna.LocationAttention(
            hidden_size=model_config.hidden_size,
            nfilters=model_config.attn_filters,
            kernel_size=model_config.attn_kernsize,
            attention_smoothing=model_config.attn_smoothing,
        )
        self.decoder = rnnd.RNNDecoder(
            hidden_size=model_config.hidden_size,
            embedding_size=model_config.embedding_size,
            vocab_size=model_config.output_units,
            nlayers=model_config.decoder_layers,
            attention=attention,
            dropout=model_config.dropout,
            tradeoff_context_embed=model_config.tradeoff_context_embedding,
            multinomial=model_config.multinomial,
        )
        self.encoder_proj = nn.Linear(
            model_config.hidden_size,
            model_config.hidden_size,
        )
        self.img_width, self.img_height = data_config.target_shape
        self.output_max_len = data_config.target_seqlen
        self.vocab_size = model_config.output_units
        self.teacher_rate = model_config.teacher_rate

    def forward(
        self,
        src: TensorType,
        tar: TensorType,
        src_len: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Transcribes the contents of an image.

        Parameters
        ----------
        src : TensorType
            A batched set of input images. A tensor of shape N x C x H x W where N is
            the batch size, C is the number of channels, H is the height and W is the
            width of the image.
        tar : TensorType
            A batched set of transcriptions. A tensor of shape N x S where N is the
            batch size and S is the sequence length. It contains the list of indices
            representing each character of the image.
        src_len : TensorType
            A batched set of input lengths. It has shape N (batch size) and it contains
            the width of the input images without accounting for padding.

        Returns
        -------
        Tuple[TensorType, List[TensorType]]:
            The set of output transcriptions in an N x S - 1 tensor where N is the batch
            size and S is the sequence length and the set of attention weights packed
            in a list of length S and contained in tensors of shape N x S.
        """
        batch_size, _, _, _ = src.shape
        tar = tar.permute(1, 0)
        # (seqlen, batch)

        src_len = torch.ceil(src_len / 16).to(torch.int)
        src_len = torch.maximum(
            torch.ones(src_len.shape[0], dtype=torch.int).to(src_len.device),
            src_len,
        )

        out_enc, hidden_enc = self.encoder(src, src_len)
        # Output: (seqlen, batch, hidden)
        # Hidden: (nlayers, batch, hidden)
        attn_proj = self.encoder_proj(out_enc)
        # (seqlen, batch, hidden)
        attn_proj = attn_proj.permute(1, 0, 2)
        # (batch, seqlen, hidden)

        attn_weights = Variable(
            torch.zeros(batch_size, self.img_width // 16),
            requires_grad=True,
        ).to(out_enc.device)
        # (batch, seqlen)

        outputs = Variable(
            torch.zeros(self.output_max_len - 1, batch_size, self.vocab_size),
            requires_grad=True,
        )
        output = Variable(one_hot(tar[0].data))
        attns = []
        outputs = outputs.cuda()
        hidden = hidden_enc

        for t in range(0, self.output_max_len - 1):  # max_len: groundtruth + <END>
            teacher_force_rate = random.random() < self.teacher_rate
            output, hidden, attn_weights = self.decoder(
                output,
                hidden,
                out_enc,
                attn_proj,
                src_len,
                attn_weights,
            )
            outputs[t] = output

            output = Variable(
                one_hot(tar[t + 1].data)
                if self.train and teacher_force_rate
                else output.data
            )
            attns.append(attn_weights.data.detach().cpu())

        return outputs, attns
