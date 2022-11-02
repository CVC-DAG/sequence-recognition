from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple

from data.generic_decrypt import GenericSample


class ModelOutput(NamedTuple):
    ...


class AlignerBase(ABC):
    def __init__(
            self
    ) -> None:
        super(ABC).__init__()

    @abstractmethod
    def evaluate_model(
            self,
    ) -> None:
        ...

    @abstractmethod
    def train_model(
            self,
    ) -> None:
        ...

    @abstractmethod
    def _inference(
            self,
    ) -> ModelOutput:
        ...


class Evaluator:
    def __init__(
            self
    ) -> None:
        pass


