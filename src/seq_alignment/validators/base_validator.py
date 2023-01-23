from pathlib import Path
from typing import Optional, Tuple

from torch import nn
from torch.utils import data as D

from ..formatters.base_formatter import BaseFormatter
from ..metrics.base_metric import BaseMetric


class BaseValidator:
    def __init__(
            self,
            valid_data: D.Dataset,
            valid_formatter: BaseFormatter,
            valid_metric: BaseMetric,
            save_path: Path,
            batch_size: int,
            workers: int,
    ) -> None:
        self.valid_data = D.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=workers,
        )
        self.valid_formatter = valid_formatter
        self.save_path = save_path

    def maximise(self) -> bool:
        raise NotImplementedError

    def validate(
            self,
            model: nn.Module,
            epoch: int
    ) -> Tuple[float, float]:
        raise NotImplementedError
