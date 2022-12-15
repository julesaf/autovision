from abc import ABC, abstractmethod

from typing import List

from core.dataset import Dataset
from core.batch import Batch
from core.sample.sample import Sample


class Model(ABC):
    @abstractmethod
    def __init__(self, train_dataset: Dataset = None, valid_dataset: Dataset = None,
                 model_ckpt_filepath=None, cat_to_label: dict = None) -> None:
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model_ckpt_filepath = model_ckpt_filepath
        self.cat_to_label = cat_to_label

    @abstractmethod
    def configure_optimizers(self, n_training_steps, lr: float = 0.001) -> None:
        pass

    @abstractmethod
    def train(self, n_epochs=1, batch_size=32, evaluate_train_set=True) -> None:
        pass

    @abstractmethod
    def _train_step(self, batch: Batch):
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self, samples: List[Sample]) -> list:
        pass
