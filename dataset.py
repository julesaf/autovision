from __future__ import annotations
from typing import Generator
import os
import json
import random
from .batch import BatchWrapper, Batch
from .sample import Sample
from .bbox import Bbox


class Dataset:
    def __init__(self, samples: list[Sample] = None) -> None:
        # TODO : utiliser torch.Dataset ou autre ?
        self.samples = []
        self.labeling_data_dirpath = None
        self.imgs_dirpath = None
        self.cat_to_label = {}
        self.n_classes = None
        self.infos_data = None
        if samples is not None:
            assert (all(isinstance(sample, Sample) for sample in samples))
            self.samples = samples
            self.cat_to_label = {}
            self.n_classes = None
            self.infos_data = None

    def from_dirs(self, labeling_data_dirpath: str, imgs_dirpath: str, mode: int = 1) -> None:
        # TODO : à migrer vers utils
        self.labeling_data_dirpath = labeling_data_dirpath
        self.imgs_dirpath = imgs_dirpath
        self._load_labeling_data()
        self._generate_samples(mode=mode)

    def get_batch(self, batch_size: int = 32, size: tuple[int, int] = None, mode: int = 1) -> Batch:
        assert (batch_size >= 1)
        batch_samples = []
        sample_generator = self._get_sample_generator()
        for _ in range(batch_size):
            sample = next(sample_generator)
            batch_samples.append(sample)
        return BatchWrapper(self.cat_to_label, batch_samples).get(size=size, mode=mode)

    def _generate_samples(self, mode: int = 1) -> None:
        self.samples = []
        for img_data in self._imgs_data:
            bboxes = []
            for bbox_id, annotation in enumerate(self._annotations_data):
                if annotation['image_id'] == img_data['id']:
                    bbox = Bbox(
                        bbox_id,
                        annotation['bbox'],
                        self.cat_to_label[annotation['category_id']],
                        mode=mode
                    )
                    bboxes.append(bbox)
            sample = Sample(
                _id=img_data['id'],
                img=os.path.join(self.imgs_dirpath, img_data['file_name']),
                bboxes=bboxes,
                img_size=(img_data['height'], img_data['width']),
                mode=mode
            )
            self.samples.append(sample)

    def _load_labeling_data(self) -> None:
        with open(self.labeling_data_dirpath) as f:
            labeling_data = json.load(f)
        self._imgs_data = labeling_data['images']
        self._cats_data = labeling_data['categories']
        self._annotations_data = labeling_data['annotations']
        self.infos_data = labeling_data['info']
        self.cat_to_label = {cat['id']: cat['name'] for cat in self._cats_data}
        self.n_classes = len(self.cat_to_label)

    def _get_sample_generator(self) -> Generator[Sample]:
        while True:
            random.shuffle(self.samples)
            for i, sample in enumerate(self.samples):
                yield sample

    def __repr__(self) -> str:
        return f"{self.infos_data}"

    def __len__(self) -> int:
        return len(self.samples)
