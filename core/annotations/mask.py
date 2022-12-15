from __future__ import annotations
import numpy as np
import cv2


class Mask:
    def __init__(self, _id, heatmap: np.ndarray, label: str) -> None:
        self.id = _id
        self.heatmap = heatmap
        self.label = label

    def get_resized(self, dsize: tuple[int, int]) -> Mask:
        resized_heatmap = cv2.resize(self.heatmap, dsize=dsize, interpolation=cv2.INTER_AREA)
        return Mask(self._id, resized_heatmap, self.label)
