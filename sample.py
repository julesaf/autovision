from __future__ import annotations

import PIL
import cv2
import os
import numpy as np
import copy
from typing import Union, Literal
from .bbox import Bbox
from .utils import visualize_bboxes, show_img


class Sample:
    def __init__(self, _id, img: Union[str, np.ndarray],
                 bboxes: list[Bbox] = None, img_size: tuple[int, int] = None,
                 mode=1) -> None:
        self.id = _id
        self.img_size = img_size
        if self.img_size is None:
            self._get_img_size()
        self.bboxes = []
        if bboxes is not None:
            self.bboxes = bboxes
        self._mode = mode
        self.is_bboxes_normalized = False
        self.img_filepath = None
        self.img = None
        if isinstance(img, str):
            assert(os.path.isfile(img), "File does not exist or check the path")
            self.img_filepath = img
        else:
            assert (isinstance(img, np.ndarray), "Must be in numpy format")
            assert (img.shape[-1] == 3, "Must be channel last format")
            self.img = img

    def visualize(self, figsize: tuple[int, int] = (12, 10)) -> None:
        if self.bboxes is not None:
            img = self.get_img()
            self.mode = 1
            visualize_bboxes(
                img,
                [bbox.coord for bbox in self.bboxes],
                [bbox.label for bbox in self.bboxes],
                figsize=figsize
            )
        else:
            img = self.get_img(img_format='channel_last')
            show_img(img, figsize=figsize)

    def get_img(self, size: tuple[int, int] = None,
                img_format: Literal['channel_first', 'channel_last'] = 'channel_first') -> np.array:
        img = self.img
        if self.img is None:
            img = np.array(PIL.Image.open(self.img_filepath))
        if size is not None:
            img = cv2.resize(img, (size[1], size[0]))
        else:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        assert img_format in ['channel_first', 'channel_last']
        if img_format != 'channel_last':
            img = img.transpose((2, 0, 1))
        return img

    def get(self, size: tuple[int, int] = None, mode: int = None) -> Sample:
        bboxes = copy.deepcopy(self.bboxes)
        if bboxes is not None:
            if size:
                x_scale = size[1] / self.img_size[1]
                y_scale = size[0] / self.img_size[0]
                bboxes = self._get_resized_bboxes(self.bboxes, x_scale, y_scale)
            else:
                size = self.img_size
            if mode is not None:
                self.mode = mode
        return Sample(self.id, self.get_img(img_format='channel_last'), img_size=size, bboxes=bboxes, mode=self._mode)

    def normalize_bboxes(self, img_size: tuple[int, int]) -> None:
        for bbox in self.bboxes:
            bbox.normalize_coord(img_size)
        self.is_bboxes_normalized = True

    def _get_img_size(self):
        self.img_size = self.img.shape[:-1]

    @staticmethod
    def _get_resized_bboxes(bboxes: list[Bbox], x_scale: float, y_scale: float) -> list[Bbox]:
        new_bboxes = []
        for bbox in bboxes:
            new_bbox = bbox.get_resized(x_scale, y_scale)
            new_bboxes.append(new_bbox)
        return new_bboxes

    @property
    def mode(self):
        if self._mode == 1:
            return "mode: (xmin, ymin, xmax, ymax)"
        elif self._mode == 2:
            return "mode: (xmin, ymin, height, width)"

    @mode.setter
    def mode(self, mode: int) -> None:
        for bbox in self.bboxes:
            bbox.mode = mode
        self._mode = mode

    def __repr__(self) -> str:
        return f"""Sample {self.id} :
        --> filepath: {self.img_filepath}
        --> n_bboxes: {len(self.bboxes)}
        """
