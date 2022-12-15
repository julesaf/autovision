from __future__ import annotations
import numpy as np


class Bbox:
    def __init__(self, _id, coord: list[int, int, int, int], label: str, box_format: int = 1) -> None:
        self.id = _id
        self.coord = coord
        self.label = label
        self.is_coord_normalized = False
        self._area = (coord[2] - coord[0]) * (coord[3] - coord[1])
        self._box_format = box_format

    def get_resized(self, x_scale: float, y_scale: float) -> Bbox:
        box_format = self._box_format
        self.box_format = 1
        orig_xmin, orig_ymin, orig_xmax, orig_ymax = self._coord
        xmin = np.round(orig_xmin * x_scale, decimals=0).astype(int)
        ymin = np.round(orig_ymin * y_scale, decimals=0).astype(int)
        xmax = np.round(orig_xmax * x_scale, decimals=0).astype(int)
        ymax = np.round(orig_ymax * y_scale, decimals=0).astype(int)
        new_coord = [xmin, ymin, xmax, ymax]
        bbox = Bbox(self.id, new_coord, self.label, box_format=1)
        if box_format != 1:
            self.box_format = box_format
            bbox.box_format = box_format
        return bbox

    def normalize_coord(self, img_size: tuple[int, int]) -> None:
        if not self.is_coord_normalized:
            self._coord[0] /= img_size[1]
            self._coord[1] /= img_size[0]
            if self._box_format == 1:
                self._coord[2] /= img_size[1]
                self._coord[3] /= img_size[0]
            else:
                self._coord[2] /= img_size[0]
                self._coord[3] /= img_size[1]
        self.is_coord_normalized = True

    def _change_to_xyxy_format(self) -> None:
        if self._box_format != 1:
            xmin, ymin, w, h = self._coord
            self.coord = (xmin, ymin, xmin + w, ymin + h)
            self._box_format = 1

    def _change_to_xywh_format(self) -> None:
        if self._box_format != 2:
            xmin, ymin, xmax, ymax = self._coord
            self.coord = (xmin, ymin, xmax - xmin, ymax - ymin)
            self._box_format = 2

    @property
    def box_format(self):
        if self._box_format == 1:
            return "box_format: (xmin, ymin, xmax, ymax)"
        elif self._box_format == 2:
            return "box_format: (xmin, ymin, height, width)"

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

    @box_format.setter
    def box_format(self, new_box_format: int):
        if new_box_format == 1 and self._box_format != 1:
            self._change_to_xyxy_format()
            self._box_format = 1
        elif new_box_format == 2 and self._box_format != 2:
            self._change_to_xywh_format()
            self._box_format = 2

    def __repr__(self) -> str:
        return f"\n{self.id} - {self.label} - {self.coord}"
