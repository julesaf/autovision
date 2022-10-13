from __future__ import annotations

import numpy as np


class Bboxes:
    def __init__(self, bboxes: list[Bbox], mode: int = 1) -> None:
        self._bboxes = bboxes
        self.mode = mode
        self.is_coords_normalized = False
        self._extract()

    def get_resized(self, x_scale: float, y_scale: float) -> Bboxes:
        new_bboxes = []
        for bbox in self._bboxes:
            new_bbox = bbox.get_resized(x_scale, y_scale)
            new_bboxes.append(new_bbox)
        return Bboxes(new_bboxes, mode=self._mode)

    def normalize_coords(self, img_size: tuple[int, int]) -> None:
        for bbox in self._bboxes:
            bbox.normalize_coord(img_size)
        self.is_coords_normalized = True

    def _extract(self):
        self._coords = [bbox.coord for bbox in self._bboxes]
        self._areas = [bbox.area for bbox in self._bboxes]
        self._labels = [bbox.label for bbox in self._bboxes]

    @property
    def coords(self):
        return self._coords

    @property
    def areas(self):
        return self._areas

    @property
    def labels(self):
        return self._labels

    @property
    def mode(self):
        if self._mode == 1:
            return "mode: (xmin, ymin, xmax, ymax)"
        elif self._mode == 2:
            return "mode: (xmin, ymin, height, width)"

    @mode.setter
    def mode(self, mode: int) -> None:
        for bbox in self._bboxes:
            bbox.mode = mode
        self._mode = mode
        self._extract()

    def __len__(self) -> int:
        return len(self._bboxes)


class Bbox:
    def __init__(self, _id, coord: tuple[int, int, int, int], label: str, mode: int = 1) -> None:
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
        new_coord = (xmin, ymin, xmax, ymax)
        bbox = Bbox(self.id, new_coord, self.label)
        if mode != 1:
            bbox.mode = mode
        return bbox

    def normalize_coord(self, img_size: tuple[int, int]) -> None:
        if self.is_coord_normalized is False:
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
    def coord(self, new_coord: tuple[int, int, int, int]):
        assert(len(new_coord) == 4)
        self._coord = new_coord

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
