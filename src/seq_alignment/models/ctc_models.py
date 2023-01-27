"""Module containing all models that implement a CTC Loss."""

import torch
from torch import nn

from .base_model import BaseModel as BaseInferenceModel, BaseModelConfig
from .model_zoo import ModelZoo
from .cnns import create_resnet, BaroCNN, RESNET_EMBEDDING_SIZES
from ..data.generic_decrypt import DataConfig, BatchedSample


class CTCModel(BaseInferenceModel):
    """Model that uses a CTC loss."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Construct a model with a CTC Loss."""
        super().__init__(cfg)
        self.loss = nn.CTCLoss()

    def compute_batch(
            self, batch: BatchedSample, device: torch.device
    ) -> torch.Tensor:
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
        output = self(batch.img.to(device))

        return output

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

        Returns
        -------
        torch.float32
            The model's loss for the given input.
        """
        columns = output.shape[0]
        target_shape = batch.img[0].shape[-1]
        input_lengths = batch.curr_shape[0] * (columns / target_shape)
        input_lengths = input_lengths.numpy().astype(int).tolist()

        batch_loss = self.loss(
            output,
            batch.gt.to(device),
            input_lengths,
            tuple(batch.og_len.tolist()),
        )
        return batch_loss


class FullyConvCTCConfig(BaseModelConfig):
    """Configuration for a Fully Convolutional CTC Model."""

    width_upsampling: int
    kern_upsampling: int
    intermediate_units: int
    output_units: int
    resnet_type: int
    pretrained: bool = True


@ModelZoo.register_model
class FullyConvCTC(CTCModel):
    """A fully convolutional CTC model with convolutional upsampling."""

    MODEL_CONFIG = FullyConvCTCConfig

    def __init__(
            self,
            model_config: FullyConvCTCConfig,
            data_config: DataConfig
    ) -> None:
        """Initialise FullyConv model from parameters.

        Parameters
        ----------
        config: FullyConvCTCConfig
            Configuration object for the model.
        data_config: DataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config)

        self._model_config = model_config
        self._data_config = data_config
        self._backbone = create_resnet(
            self._model_config.resnet_type,
            self._model_config.pretrained
        )
        self._pooling = nn.AdaptiveAvgPool2d((1, None))

        self._upsample = nn.ConvTranspose2d(
            in_channels=RESNET_EMBEDDING_SIZES[self._model_config.resnet_type],
            out_channels=self._model_config.intermediate_units,
            kernel_size=(1, self._model_config.kern_upsampling),
            stride=(1, self._model_config.width_upsampling),
        )
        self._activation = nn.ReLU()
        self._output = nn.Conv2d(
            kernel_size=1,
            in_channels=self._model_config.intermediate_units,
            out_channels=self._model_config.output_units,
        )
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        # x: N x 3   x H       x W
        x = self._backbone(x)       # N x 512 x H // 32 x W // 32
        x = self._pooling(x)        # N x 512 x 1       x W // 32
        x = self._upsample(x)       # N x INT x 1       x ~(W // 32 - 1) * K
        x = self._activation(x)
        x = self._output(x)         # N x CLS x~(W // 32 - 1) * K
        x = x.squeeze(2)            # N x INT x ~(W // 32 - 1) * K
        x = x.permute((2, 0, 1))    # ~(W // 32 - 1) * K x N x  CLS
        y = self._softmax(x)

        return y

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

    def compute_loss(
        self, batch: BatchedSample, output: torch.Tensor
    ) -> torch.float32:
        """Generate the model's loss for a single input batch and output.

        Parameters
        ----------
        batch_in: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        output: torch.Tensor
            The output of the model for the input batch.

        Returns
        -------
        torch.float32
            The model's loss for the given input.
        """
        raise NotImplementedError


class BaroCRNNConfig(BaseModelConfig):
    """Configuration for the Baró CTC Model."""

    lstm_hidden_size: int
    lstm_layers: int
    blstm: bool
    dropout: float
    output_classes: int


@ModelZoo.register_model
class BaroCRNN(CTCModel):
    """CRNN Model based on Arnau Baró's CTC OMR model."""

    MODEL_CONFIG = BaroCRNNConfig

    def __init__(
            self,
            model_config: BaroCRNNConfig,
            data_config: DataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        config: BaroCRNNConfig
            Configuration object for the model.
        data_config: DataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config)

        self.model_config = model_config
        self.data_config = data_config

        self.directions = 2 if self.model_config.blstm else 1

        self.backbone = BaroCNN(self.model_config.dropout)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=self.model_config.lstm_layers,
            dropout=self.model_config.dropout,
            bidirectional=self.model_config.blstm,
        )
        self.linear = nn.Linear(
            self.model_config.lstm_hidden_size,
            self.model_config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        x = self.backbone(x)
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape

            x = x.view(
                seq_len,
                batch_size,
                self.directions,
                hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.linear(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x


class ResnetCRNNConfig(BaseModelConfig):
    """Configuration for the Baró CTC Model."""

    resnet_type: int
    lstm_layers: int
    lstm_hidden_size: int
    upsampling_kern: int
    upsampling_stride: int
    blstm: bool
    dropout: float
    output_classes: int


@ModelZoo.register_model
class ResnetCRNN(CTCModel):
    """CRNN Model with a ResNet as backcbone."""

    MODEL_CONFIG = ResnetCRNNConfig

    def __init__(
            self,
            model_config: ResnetCRNNConfig,
            data_config: DataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        config: ResnetCRNNConfig
            Configuration object for the model.
        data_config: DataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config)

        self.model_config = model_config
        self.data_config = data_config

        self.directions = 2 if self.model_config.blstm else 1
        self.hidden_size = RESNET_EMBEDDING_SIZES[self.model_config.resnet_type]

        self.backbone = create_resnet(
            self.model_config.resnet_type,
            headless=True
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.upsample = nn.ConvTranspose2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=(1, self.model_config.upsampling_kern),
            stride=(1, self.model_config.upsampling_stride),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=self.model_config.lstm_layers,
            bidirectional=self.model_config.blstm,
            dropout=self.model_config.dropout,
        )
        self.output_layer = nn.Linear(
            self.model_config.lstm_hidden_size,
            self.model_config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        x = self.backbone(x)    # Batch, Channels, Height, Width // 16
        x = self.avg_pool(x)    # Batch, Channels, 1, Width // 16
        x = self.upsample(x)    # Batch, Channels, 1, Length
        x = x.squeeze(2)        # Batch, Channels, Length
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)     # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape
            x = x.view(
                seq_len,
                batch_size,
                self.directions, hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.output_layer(x)      # Length, Batch, Classes
        x = self.log_softmax(x)

        return x
