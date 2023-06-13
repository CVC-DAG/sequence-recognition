"""RNN-based decoder modules."""

from typing import Optional, Tuple

from torch import nn
from torch import TensorType
import torch


MULTINOMIAL = False


class RNNDecoder(nn.Module):
    """Implements an RNN-based decoder."""

    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        vocab_size: int,
        nlayers: int,
        attention: nn.Module,
        dropout: float = 0.5,
        tradeoff_context_embed: Optional[float] = None,
        multinomial: bool = False,
    ) -> None:
        """Construct Decoder.

        Parameters
        ----------
        hidden_size : int
            Dimensionality of the hidden states of the gated recurrent unit stack.
        embedding_size : int
            Size of the token embeddings.
        vocab_size : int
            Size of the model's vocabulary.
        nlayers : int
            Number of layers in the gated recurrent unit stack.
        attention : nn.Module
            Attention block module.
        dropout : float, optional
            Amount of dropout to apply on the gated recurrent units, by default 0.5.
        tradeoff_context_embed : Optional[float], optional
            By how much to reduce the context embedding dimension, by default None
        multinomial : bool, optional
            Whether to change the input character by a multinomial distribution sampling
            version of it, by default False.
        """
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.nlayers = nlayers
        self.tradeoff = tradeoff_context_embed
        self.multinomial = multinomial
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.attention = attention
        if self.tradeoff is not None:
            tradeoff_size = int(self.embed_size * self.tradeoff)
            self.context_shrink = nn.Linear(self.hidden_size, tradeoff_size)
            self.gru = nn.GRU(
                tradeoff_size + self.embed_size,
                self.hidden_size,
                self.nlayers,
                dropout=self.dropout,
            )
        else:
            self.gru = nn.GRU(
                self.embed_size + self.hidden_size,
                self.hidden_size,
                self.nlayers,
                dropout=self.dropout,
            )
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def _compute_logits(
        self,
        in_char: TensorType,
        hidden: TensorType,
        encoder_output: TensorType,
        attn_proj: TensorType,
        src_len: TensorType,
        prev_attn: TensorType,
    ) -> Tuple[TensorType, TensorType, TensorType]:
        attn_weights = self.attention(hidden, attn_proj, src_len, prev_attn)
        attn_weights = attn_weights.unsqueeze(-1)
        # (batch, seqlen, 1)

        encoder_output = encoder_output.permute(1, 2, 0)
        # (batch, hidden, seqlen)
        context = torch.bmm(encoder_output, attn_weights)
        # (batch, hidden, 1)
        context = context.squeeze(2)
        # (batch, hidden)

        if self.tradeoff is not None:
            context = self.context_shrink(context)

        if self.multinomial and self.training:
            top1 = torch.multinomial(in_char, 1)
        else:
            top1 = torch.argmax(in_char, dim=1)
        embed_char = self.embedding(top1)
        # (batch, embedding)

        in_dec = torch.cat((embed_char, context), 1)
        # (batch, hidden + embedding)
        in_dec = in_dec.unsqueeze(0)
        # (1, batch, hidden + embedding) -> For sequence length
        output, latest_hidden = self.gru(in_dec, hidden.contiguous())
        # Output: (1, batch, hidden)
        # Hidden: (layers, hidden)
        output = output.squeeze(0)

        return output, latest_hidden, attn_weights.squeeze(2)

    def forward(
        self,
        in_char: TensorType,
        hidden: TensorType,
        encoder_output: TensorType,
        attn_proj: TensorType,
        src_len: TensorType,
        prev_attn: TensorType,
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """Compute a single decoding step.

        Parameters
        ----------
        in_char : TensorType
            A one-hot encoded representation of the input character.
        hidden : TensorType
            The previous hidden state of the Decoder.
        encoder_output : TensorType
            Output of the encoder of the model. It is a S x N x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        attn_proj : TensorType
            The layer-projected version of the output of the encoder. It is a N x S x H
            tensor, where N is the batch size, S is the sequence length and H is the
            hidden dimension size.
        src_len : TensorType
            Length in intermediate columns of the input image. This is used to account
            for padding.
        prev_attn : TensorType
            The attention weights of the previous iteration. It is a N x S tensor, where
            N is the batch size and S is the sequence length.

        Returns
        -------
        Tuple[TensorType, TensorType, TensorType]
            The output of the model at timestep t, the last hidden state and the
            attention weights. They are N x V, N x H and N x S tensors respectively,
            where N is the batch size, V is the vocab size, H is the hidden size and S
            is the maximum sequence length. The output contains the logit probabilities
            of each class of the model.
        """
        output, hidden, attention = self._compute_logits(
            in_char,
            hidden,
            encoder_output,
            attn_proj,
            src_len,
            prev_attn,
        )
        output = self.out(output)

        return (
            output,
            hidden,
            attention,
        )


class RNN2HeadDecoder(RNNDecoder):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        vocab_size: int,
        secondary_vocab_size: int,
        nlayers: int,
        attention: nn.Module,
        dropout: float = 0.5,
        tradeoff_context_embed: Optional[float] = None,
        multinomial: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            embedding_size,
            vocab_size,
            nlayers,
            attention,
            dropout,
            tradeoff_context_embed,
            multinomial,
        )
        self.secondary_head = nn.Linear(self.hidden_size, secondary_vocab_size)

    def forward(
        self,
        in_char: TensorType,
        hidden: TensorType,
        encoder_output: TensorType,
        attn_proj: TensorType,
        src_len: TensorType,
        prev_attn: TensorType,
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """Compute a single decoding step.

        Parameters
        ----------
        in_char : TensorType
            A one-hot encoded representation of the input character.
        hidden : TensorType
            The previous hidden state of the Decoder.
        encoder_output : TensorType
            Output of the encoder of the model. It is a S x N x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        attn_proj : TensorType
            The layer-projected version of the output of the encoder. It is a N x S x H
            tensor, where N is the batch size, S is the sequence length and H is the
            hidden dimension size.
        src_len : TensorType
            Length in intermediate columns of the input image. This is used to account
            for padding.
        prev_attn : TensorType
            The attention weights of the previous iteration. It is a N x S tensor, where
            N is the batch size and S is the sequence length.

        Returns
        -------
        Tuple[TensorType, TensorType, TensorType]
            The output of the model at timestep t, the last hidden state and the
            attention weights. They are N x V, N x H and N x S tensors respectively,
            where N is the batch size, V is the vocab size, H is the hidden size and S
            is the maximum sequence length. The output contains the logit probabilities
            of each class of the model.
        """
        output, hidden, attention = self._compute_logits(
            in_char,
            hidden,
            encoder_output,
            attn_proj,
            src_len,
            prev_attn,
        )
        main_output = self.out(output)
        secondary_output = self.secondary_head(output)

        return (
            main_output,
            secondary_output,
            hidden,
            attention,
        )
