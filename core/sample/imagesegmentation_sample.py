from __future__ import annotations

import numpy as np
import copy
import utils
from typing import Union
from core.annotations.mask import Mask
from core.sample.sample import Sample


class ImageSegmentationSample(Sample):
    def __init__(self,
                 _id,
                 img: Union[str, np.ndarray],
                 annotations: list[Mask] = None,
                 img_size: tuple[int, int] = None) -> None:
        super().__init__(_id, img, annotations, img_size)

    def visualize(self, figsize: tuple[int, int] = (12, 10)) -> None:
        if self.annotations is not None:
            img = self.get_img(img_format='channel_last')
            utils.image.visualize_masks(
                img,
                [mask.heatmap for mask in self.annotations],
                [mask.label for mask in self.annotations],
                figsize=figsize
            )
        else:
            img = self.get_img(img_format='channel_last')
            utils.image.show_img(img, figsize=figsize)

    def get(self, size: tuple[int, int] = None, mode: int = None) -> Sample:
        annotations = copy.deepcopy(self.annotations)
        if annotations is not None:
            if size:
                annotations = self._get_resized_masks(self.annotations, size)
            else:
                size = self.img_size
        return ImageSegmentationSample(self.id, self.get_img(img_format='channel_last'),
                                       img_size=size, annotations=annotations)

    @staticmethod
    def _get_resized_masks(masks: list[Mask], dsize: tuple[int, int]) -> list[Mask]:
        new_masks = []
        for mask in masks:
            new_mask = mask.get_resized(dsize)
            new_masks.append(new_mask)
        return new_masks