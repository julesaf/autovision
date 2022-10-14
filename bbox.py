from __future__ import annotations
import numpy as np


class Bbox:
    def __init__(self, _id, coord: list[int, int, int, int], label: str, mode: int = 1) -> None:
        self.id = _id
        self.coord = coord
        self.label = label
        self.is_coord_normalized = False
        self._area = (coord[2] - coord[0]) * (coord[3] - coord[1])
        self._mode = mode

    def get_resized(self, x_scale: float, y_scale: float) -> Bbox:
        mode = self._mode
        self.mode = 1
        orig_xmin, orig_ymin, orig_xmax, orig_ymax = self._coord
        xmin = np.round(orig_xmin * x_scale, decimals=0).astype(int)
        ymin = np.round(orig_ymin * y_scale, decimals=0).astype(int)
        xmax = np.round(orig_xmax * x_scale, decimals=0).astype(int)
        ymax = np.round(orig_ymax * y_scale, decimals=0).astype(int)
        new_coord = [xmin, ymin, xmax, ymax]
        bbox = Bbox(self.id, new_coord, self.label)
        if mode != 1:
            bbox.mode = mode
        return bbox

    def normalize_coord(self, img_size: tuple[int, int]) -> None:
        if not self.is_coord_normalized:
            normalized_coord = self._coord
            self._coord[0] /= img_size[1]
            self._coord[1] /= img_size[0]
            if self._mode == 1:
                self._coord[2] /= img_size[1]
                self._coord[3] /= img_size[0]
            else:
                self._coord[2] /= img_size[0]
                self._coord[3] /= img_size[1]
        self.is_coord_normalized = True

    def _change_to_mode1(self) -> None:
        if self._mode != 1:
            xmin, ymin, w, h = self._coord
            self.coord = (xmin, ymin, xmin + w, ymin + h)
            self._mode = 1

    def _change_to_mode2(self) -> None:
        if self._mode != 2:
            xmin, ymin, xmax, ymax = self._coord
            self.coord = (xmin, ymin, xmax - xmin, ymax - ymin)
            self._mode = 2

    @property
    def mode(self):
        if self._mode == 1:
            return "mode: (xmin, ymin, xmax, ymax)"
        elif self._mode == 2:
            return "mode: (xmin, ymin, height, width)"

    @property
    def coord(self):
        return self._coord

    @property
    def area(self):
        return self._area

    @coord.setter
    def coord(self, new_coord: list[int, int, int, int]):
        assert(len(new_coord) == 4)
        self._coord = list(new_coord)

    @mode.setter
    def mode(self, new_mode: int):
        if new_mode == 1 and self._mode != 1:
            self._change_to_mode1()
            self._mode = 1
        elif new_mode == 2 and self._mode != 2:
            self._change_to_mode2()
            self._mode = 2

    def __repr__(self) -> str:
        return f"\n{self.id} - {self.label} - {self.coord}"
