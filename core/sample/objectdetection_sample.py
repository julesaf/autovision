from __future__ import annotations

import numpy as np
import copy
import utils
from typing import Union
from core.annotations.bbox import Bbox
from .sample import Sample


class ObjectDetectionSample(Sample):
    def __init__(self,
                 _id,
                 img: Union[str, np.ndarray],
                 annotations: list[Bbox] = None,
                 img_size: tuple[int, int] = None,
                 box_format=1) -> None:
        super().__init__(_id, img, annotations, img_size)
        self._box_format = box_format
        # self.is_bboxes_normalized = False

    @staticmethod
    def from_prediction(img, prediction, threshold: float = 0.5):
        bboxes_preds = prediction['boxes'][prediction['scores'] > threshold]
        labels_preds = prediction['labels'][prediction['scores'] > threshold]
        annotations = [Bbox(0, bbox, label) for bbox, label in zip(bboxes_preds, labels_preds)]
        return ObjectDetectionSample(0, img, annotations)

    def visualize(self, figsize: tuple[int, int] = (12, 10)) -> None:
        if self.annotations is not None:
            img = self.get_img(img_format='channel_first')
            self.box_format = 1
            utils.image.visualize_bboxes(
                img,
                [bbox.coord for bbox in self.annotations],
                [bbox.label for bbox in self.annotations],
                figsize=figsize
            )
        else:
            img = self.get_img(img_format='channel_last')
            utils.image.show_img(img, figsize=figsize)

    def get(self, size: tuple[int, int] = None, box_format: int = None) -> Sample:
        annotations = copy.deepcopy(self.annotations)
        if annotations is not None:
            if size:
                x_scale = size[1] / self.img_size[1]
                y_scale = size[0] / self.img_size[0]
                annotations = self._get_resized_bboxes(self.annotations, x_scale, y_scale)
            else:
                size = self.img_size
            if box_format is not None:
                for annotation in annotations:
                    annotation.box_format = box_format
        return ObjectDetectionSample(self.id, self.get_img(size=size, img_format='channel_last'), img_size=size,
                                     annotations=annotations, box_format=box_format)

    # def normalize_bboxes(self, img_size: tuple[int, int]) -> None:
    #     for bbox in self.bboxes:
    #         bbox.normalize_coord(img_size)
    #     self.is_bboxes_normalized = True

    @staticmethod
    def _get_resized_bboxes(bboxes: list[Bbox], x_scale: float, y_scale: float) -> list[Bbox]:
        new_bboxes = []
        for bbox in bboxes:
            new_bbox = bbox.get_resized(x_scale, y_scale)
            new_bboxes.append(new_bbox)
        return new_bboxes

    @property
    def box_format(self):
        if self._box_format == 1:
            return "box_format: (xmin, ymin, xmax, ymax)"
        elif self._box_format == 2:
            return "box_format: (xmin, ymin, height, width)"

    @box_format.setter
    def box_format(self, box_format: int) -> None:
        for bbox in self.annotations:
            bbox.box_format = box_format
        self._box_format = box_format
