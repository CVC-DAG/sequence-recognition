from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils import data as D

from data.generic_decrypt import BatchedSample, GenericDecryptDataset
from models.types import *

from validator import Validator

Loss = object


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_data: D.DataLoader,
            optimiser: Optimizer,
            loss_func: Loss,
            clip_grad: Optional[float],
            validator: Validator,
            
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.clip_grad = clip_grad

        self.validator = validator

    def compute_loss(
            self,
            batch_in: BatchedSample,
    ) -> float:
        output = self.model(batch_in)
        loss = self.loss_func(output, batch_in.gt, 

        return loss
