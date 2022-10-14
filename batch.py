from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
import numpy as np


class BatchWrapper:
    def __init__(self, cat_to_label: dict = None, samples: list = None) -> None:
        if cat_to_label is not None:
            label_to_cat = {v: k for k, v in cat_to_label.items()}
            self.label_to_cat = lambda label: label_to_cat[label]
        self.samples = samples
        if self.samples is None:
            self.samples = []
        self._batch = None

    def append(self, sample) -> None:
        self.samples.append(sample)

    def get(self, size: tuple[int, int] = None, mode: int = 1,
            normalize_imgs: bool = False, normalize_bboxes: bool = False) -> Batch:
        if size is None:
            size = self._get_mean_size()
        self._transform_in_batch(size, mode)
        if normalize_imgs:
            if self._batch is not None:
                self._batch.normalize_imgs()
        if normalize_bboxes:
            if self._batch is not None:
                self._batch.normalize_bboxes(size=size)
        return self._batch

    def _get_mean_size(self) -> tuple[int, int]:
        mean = lambda e: np.round(np.mean(e), decimals=0).astype(int)
        img_height_mean = mean([sample.img_size[0] for sample in self.samples])
        img_width_mean = mean([sample.img_size[0] for sample in self.samples])
        return img_height_mean, img_width_mean

    def _transform_in_batch(self, size: tuple[int, int], mode: int = 1) -> None:
        self._batch = Batch()
        for sample in self.samples:
            sample = sample.get(size, mode)
            if len(sample.bboxes) > 0:
                self._batch.append(
                    sample.get_img(),
                    [bbox.coord for bbox in sample.bboxes],
                    [bbox.area for bbox in sample.bboxes],
                    [bbox.label for bbox in sample.bboxes],
                    [self.label_to_cat(bbox.label) for bbox in sample.bboxes]
                )
            else:
                self._batch.append(sample.get_img())
        self._batch.to_numpy()

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class Batch:
    pixel_values: Union[list, np.array] = field(default_factory=list)
    bboxes: Union[list, np.array] = field(default_factory=list)
    areas: Union[list, np.array] = field(default_factory=list)
    labels: Union[list, np.array] = field(default_factory=list)
    categories: Union[list, np.array] = field(default_factory=list)
    is_imgs_normalized: bool = False
    is_bboxes_normalized: bool = False

    def append(self, new_pixel_values, new_bboxes=None, new_areas=None, new_labels=None, new_categories=None):
        self.pixel_values.append(new_pixel_values)
        if new_bboxes is not None:
            self.bboxes.append(new_bboxes)
        if new_areas is not None:
            self.areas.append(new_areas)
        if new_labels is not None:
            self.labels.append(new_labels)
        if new_categories is not None:
            self.categories.append(new_categories)

    def to_numpy(self):
        self.pixel_values = np.array(self.pixel_values)
        self.bboxes = np.array(self.bboxes)
        self.areas = np.array(self.areas)
        self.labels = np.array(self.labels)
        self.categories = np.array(self.categories)

    def normalize_imgs(self) -> None:
        self.to_numpy()
        if len(self.pixel_values) > 0 and (not self.is_imgs_normalized):
            self.pixel_values = self.pixel_values / 255.
            self.is_imgs_normalized = True

    def normalize_bboxes(self, size: tuple[int, int]) -> None:
        self.to_numpy()
        if len(self.bboxes) > 0 and (not self.is_bboxes_normalized):
            self.bboxes = self.bboxes / np.array([size[1], size[0], size[1], size[0]])
            self.is_bboxes_normalized = True

    def __len__(self):
        return len(self.pixel_values)
# padding
