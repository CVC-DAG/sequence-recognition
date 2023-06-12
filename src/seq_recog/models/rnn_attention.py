from torch import nn
from torch import TensorType
import torch


class LocationAttention(nn.Module):
    """Implements Bahdanau + Location attention."""

    def __init__(
        self,
        hidden_size: int,
        nfilters: int,
        kernel_size: int,
        attention_smoothing: bool = False,
    ):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.conv1d = nn.Conv1d(1, nfilters, kernel_size, padding="same")
        self.prev_attn_proj = nn.Linear(nfilters, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if attention_smoothing:
            self.sigma = self._attn_smoothing
        else:
            self.sigma = nn.Softmax(dim=-1)

    def _attn_smoothing(self, x):
        return self.sigmoid(x) / self.sigmoid(x).sum(axis=-1)

    def forward(
        self,
        hidden: TensorType,
        encoder_output: TensorType,
        enc_len: TensorType,
        prev_attention: TensorType,
    ) -> TensorType:
        """Produce normalised attention weights for a single decoding interation.

        Parameters
        ----------
        hidden : TensorType
            Current hidden state of the decoder. It is a N x H x L tensor, where N is
            the batch size, L is the number of decoder layers and H is the hidden
            dimension size.
        encoder_output : TensorType
            Output of the encoder of the mode. It is a S x N x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        enc_len : TensorType
            A tensor containing the length in number of columns of the input batch (in
            other words, the amount of feature vectors that are not padding).
        prev_attention : TensorType
            The attention weights for the previous decoding time step. It is an N x S
            tensor, where N is the batch size and S is the sequence length.

        Returns
        -------
        TensorType
            A tensor containing the attention weights for the current time step. It is
            an N x S tensor, where N is the batch size and S is the sequence length.
        """
        encoder_output = encoder_output.transpose(0, 1)
        # (batch, seqlen, hidden)
        attn_energy = self.score(hidden, encoder_output, prev_attention)
        # (batch, seqlen)

        batch, seqlen, hidden = encoder_output.shape
        indices = torch.arange(seqlen).unsqueeze(0).expand(batch, -1)
        attn_energy = self.sigma(
            torch.where(
                indices < enc_len.unsqueeze(0),
                attn_energy,
                -torch.inf,
            )
        )
        return attn_energy

    # encoder_output: b, t, f
    def score(
        self,
        hidden: TensorType,
        encoder_output: TensorType,
        prev_attention: TensorType,
    ) -> TensorType:
        """Produce the unnormalised attention scores for the given time step.

        Parameters
        ----------
        hidden : TensorType
            Current hidden state of the decoder. It is a N x H x L tensor, where N is
            the batch size, L is the number of decoder layers and H is the hidden
            dimension size.
        encoder_output : TensorType
            Output of the encoder of the model. It is a N x S x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        prev_attention : TensorType
            The attention weights for the previous decoding time step. It is an N x S
            tensor, where N is the batch size and S is the sequence length.

        Returns
        -------
        TensorType
            Unnormalised energy values for each element along the temporal (sequence
            length) dimension. It is an N x S tensor, with N being batch size and S
            being sequence length.
        """
        hidden = hidden.permute(1, 0, 2)

        hidden = hidden.sum(dim=1)
        hidden_attn = self.hidden_proj(hidden).unsqueeze(1)
        # (batch, 1, hidden)

        prev_attention = prev_attention.unsqueeze(1)
        conv_prev_attn = self.conv1d(prev_attention)
        conv_prev_attn = conv_prev_attn.permute(0, 2, 1)
        conv_prev_attn = self.prev_attn_proj(conv_prev_attn)
        # (batch, seqlen, hidden)

        # FIXME: This is the same operation every time.
        encoder_output_attn = self.encoder_output_proj(encoder_output)
        # (batch, seqlen, hidden)

        res_attn = self.tanh(encoder_output_attn + hidden_attn + conv_prev_attn)
        out_attn = self.out(res_attn)
        # (batch, seqlen, 1)

        out_attn = out_attn.squeeze(2)
        # (batch, seqlen)

        return out_attn
