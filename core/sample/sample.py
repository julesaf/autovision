from __future__ import annotations

import PIL
import cv2
import os
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Literal


class Sample(ABC):
    def __init__(self,
                 _id,
                 img: Union[str, np.ndarray],
                 annotations: list = None,
                 img_size: tuple[int, int] = None,
                 ) -> None:
        self.id = _id
        self.img = None
        self.annotations = []
        if annotations is not None:
            self.annotations = annotations
        self.img_filepath, self.img = None, None
        self._load_img_input(img)
        self.img_size = img_size
        if self.img_size is None:
            self._get_img_size()

    @staticmethod
    @abstractmethod
    def from_prediction(img: Union[str, np.ndarray], prediction, threshold: float = 0.5):
        pass

    def get_img(self, size: tuple[int, int] = None,
                img_format: Literal['channel_first', 'channel_last'] = 'channel_last') -> np.ndarray:
        img = copy.deepcopy(self.img)
        if img is None:
            img = np.array(PIL.Image.open(self.img_filepath))
        if size is not None:
            img = cv2.resize(img, (size[1], size[0]))
        # else:
        #     img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        assert img_format in ['channel_first', 'channel_last']
        if img_format != 'channel_last':
            img = img.transpose((2, 0, 1))
        return img

    def get_labels(self) -> set:
        return set([annotation.label for annotation in self.annotations])

    @abstractmethod
    def visualize(self, figsize: tuple[int, int] = (12, 10)) -> None:
        pass

    @abstractmethod
    def get(self, size: tuple[int, int] = None) -> Sample:
        pass

    def _load_img_input(self, img) -> None:
        if isinstance(img, str):
            assert os.path.isfile(img), f"File does not exist or check the path, path: {img}"
            self.img_filepath = img
        else:
            assert isinstance(img, np.ndarray), "Must be in numpy format"
            assert img.shape[-1] == 3, "Must be channel last format"
            self.img = img

    def _get_img_size(self) -> None:
        if self.img is not None:
            self.img_size = self.img.shape[:-1]
        else:
            self.img_size = self.get_img().shape[:-1]

    def __repr__(self) -> str:
        return f"Sample number {self.id}:\n" \
               f"\t - {len(self.annotations)} annotations (included {self.get_labels()}) \n" \
               f"\t - img size --> {self.img_size}"
