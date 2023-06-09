"""RNN-based encoder modules."""

import numpy as np

from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .cnns import create_vgg, VGG_EMBEDDING_SIZE


class VggRNNEncoder(nn.Module):
    """RNN-based encoder with a VGG Backbone."""

    def __init__(
        self,
        vgg_type: int,
        vgg_bn: bool,
        vgg_pretrain: bool,
        hidden_size: int,
        layers: int,
        height: int,
        width: int,
        dropout: float,
        sum_hidden: bool = False,
    ):
        super(VggRNNEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = layers
        self.height = height
        self.width = width
        self.dropout = dropout
        self.sum_hidden = sum_hidden

        self.backbone = create_vgg(
            vgg_type=vgg_type,
            batchnorm=vgg_bn,
            headless=True,
            pretrained=vgg_pretrain,
            keep_maxpool=False,
        )
        self.backbone_dropout = nn.Dropout2d(p=dropout)

        self.rnn = nn.GRU(
            (self.height // 16) * VGG_EMBEDDING_SIZE,
            self.hidden_size,
            self.n_layers,
            dropout=self.dropout,
            bidirectional=True,
        )
        if self.sum_directions:
            self.enc_out_merge = (
                lambda x: x[:, :, : x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2 :]
            )
            self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)

    @staticmethod
    def _sum_directions(x: Tensor) -> Tensor:
        seqlen, batch, _ = x.shape
        x = x.view(seqlen, batch, 2, -1)
        return x.sum(dim=-2)

    def forward(self, in_data, in_data_len, hidden=None):
        batch_size = in_data.shape[0]
        out = self.layer(in_data)
        # (batch, channels, height, width)
        out = self.backbone_dropout(out)
        out = out.permute(3, 0, 2, 1)
        # (width, batch, height, channels)
        out = out.contiguous()
        out = out.view(-1, batch_size, self.height // 16 * 512)
        # (width, batch, channels * height)

        width = out.shape[0]
        src_len = (in_data_len.numpy() * (width / self.width)).astype("int")
        src_len = np.maximum(1, src_len)
        out = pack_padded_sequence(
            out,
            src_len.tolist(),
            batch_first=False,
            enforce_sorted=False,
        )
        output, hidden = self.rnn(out, hidden)
        output, output_len = pad_packed_sequence(
            output,
            batch_first=False,
        )
        # Output: (width, batch, 2 * hidden)
        # Hidden: (2 * n_layers, batch, hidden)

        output = self._sum_directions(output)
        # (width, batch, hidden)

        if self.sum_hidden:
            final_hidden = hidden[1::2] + hidden[0::2]
        else:
            final_hidden = hidden[1::2]
        return output, final_hidden
