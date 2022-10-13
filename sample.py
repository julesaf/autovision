from __future__ import annotations

import PIL
import cv2
import numpy as np
import copy
from typing import Union
from .bbox import Bboxes, Bbox
from .utils import visualize_bboxes, show_img


class Sample:
    def __init__(self,
                 _id,
                 img: Union[str, np.ndarray],
                 bboxes: Union[list[Bbox]] = None,
                 img_size: tuple[int, int] = None,
                 mode=1
                 ) -> None:
        self.id = _id
        self.img_filepath = None
        self.img = None
        if isinstance(img, str):
            self.img_filepath = img
        else:
            assert (isinstance(img, np.ndarray))
            assert (img.shape[-1] == 3)
            self.img = img
        self.img_size = img_size
        if self.img_size is None:
            self._get_img_size()
        self.bboxes = None
        if bboxes is not None:
            if isinstance(bboxes, Bboxes):
                self.bboxes = bboxes
            else:
                self.bboxes = Bboxes(bboxes, mode=mode)

    def visualize(self, figsize: tuple[int, int] = (12, 10)) -> None:
        img = self.get_img()
        if self.bboxes is not None:
            self.bboxes.mode = 1
            visualize_bboxes(img, self.bboxes.coords, self.bboxes.labels, figsize=figsize)
        else:
            img = img.transpose(1, 2, 0)
            show_img(img, figsize=figsize)

    def get_img(self, size: tuple[int, int] = None, img_format: str = 'channel_first') -> np.array:
        img = copy.copy(self.img)
        if self.img is None:
            img = np.array(PIL.Image.open(self.img_filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size is not None:
            img = cv2.resize(img, (size[1], size[0]))
        else:
            if self.img_size is not None:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        if img_format != 'channel_last':
            img = img.transpose(2, 0, 1)
        return img

    def get(self, size: tuple[int, int] = None, mode: int = 1) -> Sample:
        bboxes = copy.deepcopy(self.bboxes)
        if bboxes is not None:
            if size:
                x_scale = size[1] / self.img_size[1]
                y_scale = size[0] / self.img_size[0]
                bboxes = bboxes.get_resized(x_scale, y_scale)
            else:
                size = self.img_size
            bboxes.mode = mode
        return Sample(self.id, self.get_img(img_format='channel_last'), img_size=size, bboxes=bboxes)

    def _get_img_size(self):
        self.img_size = self.img.shape[:-1]

    def __repr__(self) -> str:
        return f"""Sample {self.id} :
        --> filepath: {self.img_filepath}
        --> n_bboxes: {len(self.bboxes)}
        """
