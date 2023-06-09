"""Transformer-based modules."""

from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from vit_pytorch import ViT, Extractor

from .misc import TokenEmbedding, PositionalEncoding


Width = int
Height = int
ImgShape = Tuple[Height, Width]


class ViTTransformer(nn.Module):
    """Implements a full seq2seq transformer using vit_pytorch."""

    ACTIVATIONS = {
        "relu": F.relu,
        "gelu": F.gelu,
    }

    def __init__(
        self,
        img_shape: ImgShape,
        patch_size: int,
        vocab_size: int,
        model_dim: int,
        enc_layers: int,
        dec_layers: int,
        enc_heads: int,
        dec_heads: int,
        mlp_dim: int,
        emb_dropout: float,
        dropout: float,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
        max_inference_len: int = 1000,
        norm_first: bool = False,
        activation: str = "relu",
    ) -> None:
        super(ViTTransformer, self).__init__()
        self.freeze_backbone = freeze_backbone

        self.embedder = TokenEmbedding(
            vocab_size,
            model_dim,
        )
        vit = ViT(
            image_size=img_shape,
            patch_size=patch_size,
            num_classes=vocab_size,
            dim=model_dim,
            depth=enc_layers,
            heads=enc_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        if pretrained_backbone is not None:
            weights = torch.load(pretrained_backbone)
            del weights["mlp_head.1.weight"]
            del weights["mlp_head.1.bias"]
            missing, unexpected = vit.load_state_dict(weights, strict=False)

            if missing or unexpected:
                warn(
                    f"There are missing or unexpected weights in the loaded"
                    f"encoder model: \n{'='*50}\nMissing: {missing}\n{'='*50}\n"
                    f"Unexpected:{unexpected}"
                )

        self.encoder = Extractor(vit)

        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pos_encoding = PositionalEncoding(model_dim, emb_dropout)
        self.max_inference_len = max_inference_len
        self.vocab_size = vocab_size
        self.norm = nn.LayerNorm(model_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=dec_heads,
            dim_feedforward=model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=self.ACTIVATIONS[activation],
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dec_layers, norm=self.norm
        )
        self.linear = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, y_mask, pad_mask) -> Tensor:  # x: Batch,
        _, x = self.encoder(x)  # Batch, Tok + 1, Dim

        if self.training:
            y = self.embedder(y)  # Batch, Seqlen, Dim
            y = self.pos_encoding(y)  # Batch, Seqlen, Dim

            # Embed the target sequence and positionally encode it
            x = self.decoder(y, x, tgt_mask=y_mask, tgt_key_padding_mask=pad_mask)
            x = self.linear(x)  # Batch, Seqlen, Class
            # x = self.softmax(x)
        else:
            newy = torch.zeros(y.shape[0], y.shape[1] + 1, *y.shape[2:]).to(y.device)
            newy[:, 0] = y[:, 0]
            y = newy

            outputs = torch.zeros(
                y.shape[0], self.max_inference_len - 1, self.vocab_size
            ).to(y.device)
            for ii in range(1, self.max_inference_len):
                y_t = y[:, :ii]
                y_t = self.embedder(y_t)
                y_t = self.pos_encoding(y_t)
                out = self.decoder(y_t, x, tgt_mask=y_mask[:ii, :ii])
                out = self.linear(out)

                outputs[:, ii - 1, :] = out[:, ii - 1, :]
                y[:, ii] = torch.argmax(out, -1)[:, ii - 1]

            x = outputs
        return x
