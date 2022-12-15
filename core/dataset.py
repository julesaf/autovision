from __future__ import annotations
from typing import Generator

import os
import copy
import json
import random
import cv2
import numpy as np
from typing import Literal
from .batch import BatchWrapper, Batch
from .sample.sample import Sample
from .sample.objectdetection_sample import ObjectDetectionSample
from .sample.imagesegmentation_sample import ImageSegmentationSample
from .annotations.bbox import Bbox
from .annotations.mask import Mask


class Dataset:
    def __init__(self,
                 modeltype: Literal['object_detection', 'image_segmentation'],
                 samples: list[Sample] = None
                 ) -> None:
        self._is_modeltype_set = False
        self.modeltype = modeltype
        self.samples = []
        self._labeling_data_dirpath = None
        self._imgs_dirpath = None
        self.cat_to_label = {}
        self.n_classes = None
        if samples is not None:
            assert (all(isinstance(sample, Sample) for sample in samples))
            all_labels = []
            for sample in samples:
                labels = sample.get_labels()
                all_labels.extend(list(labels))
            all_labels = set(all_labels)
            self.samples = samples
            self.cat_to_label = {}
            self.cat_to_label = {i+1: label for i, label in enumerate(all_labels)}
            self.cat_to_label[0] = 'background'
            self.n_classes = len(self.cat_to_label)

    def from_coco_format(self, labeling_data_dirpath: str, imgs_dirpath: str, box_format: int = 1) -> None:
        self._labeling_data_dirpath = labeling_data_dirpath
        self._imgs_dirpath = imgs_dirpath
        self._load_labeling_data()
        if self._modeltype == 'object_detection':
            self._generate_objectdetection_samples(box_format=box_format)
        elif self._modeltype == 'image_segmentation':
            self._generate_imagesegmentation_samples()

    def get_batch(self, batch_size: int = 32, size: tuple[int, int] = None, box_format: int = 1) -> Batch:
        assert (batch_size >= 1)
        batch_samples = []
        sample_generator = self._get_sample_generator()
        for _ in range(batch_size):
            sample = next(sample_generator)
            batch_samples.append(sample)
        return BatchWrapper(self.cat_to_label, batch_samples).get(size=size, box_format=box_format)

    def get_mean_size(self):
        return np.mean([sample.img_size for sample in self.samples], axis=0)

    def split(self, ratio_dataset1=0.8, seed=42):
        n_samples = len(self.samples)
        n_samples_dataset1 = int(ratio_dataset1 * n_samples) # inf
        random.seed(seed)
        random.shuffle(self.samples)
        samples_dataset1, samples_dataset2 = self.samples[:n_samples_dataset1], self.samples[n_samples_dataset1:]
        return Dataset(self._modeltype, samples_dataset1), Dataset(self._modeltype, samples_dataset2)

    def _generate_imagesegmentation_samples(self) -> None:
        self.samples = []
        for img_data in self._imgs_data:
            masks = []
            cat_heatmap = {category_id: np.zeros((int(img_data['height']), int(img_data['width']), 1))
                           for category_id in self.cat_to_label.keys()}
            for annotation_id, annotation in enumerate(self._annotations_data):
                if annotation['image_id'] == img_data['id']:
                    contours = np.array(annotation['segmentation']).round().astype(int).reshape(-1, 2)
                    cat_heatmap[annotation['category_id']+1] = cv2.fillPoly(cat_heatmap[annotation['category_id']],
                                                                          pts=[contours], color=(1))
            for cat, heatmap in cat_heatmap.items():
                mask = Mask(cat, heatmap, self.cat_to_label[cat])
                masks.append(mask)
            sample = ImageSegmentationSample(
                _id=img_data['id'],
                img=os.path.join(self._imgs_dirpath, img_data['file_name']),
                annotations=masks,
                img_size=(img_data['height'], img_data['width'])
            )
            self.samples.append(sample)

    def _generate_objectdetection_samples(self, box_format: int = 1) -> None:
        self.samples = []
        for img_data in self._imgs_data:
            bboxes = []
            for bbox_id, annotation in enumerate(self._annotations_data):
                if annotation['image_id'] == img_data['id']:
                    bbox = Bbox(
                        bbox_id,
                        annotation['bbox'],
                        self.cat_to_label[annotation['category_id']+1],
                        box_format=box_format
                    )
                    bboxes.append(bbox)
            sample = ObjectDetectionSample(
                _id=img_data['id'],
                img=os.path.join(self._imgs_dirpath, img_data['file_name']),
                annotations=bboxes,
                img_size=(img_data['height'], img_data['width']),
                box_format=box_format
            )
            self.samples.append(sample)

    def _load_labeling_data(self) -> None:
        with open(self._labeling_data_dirpath) as f:
            labeling_data = json.load(f)
        self._imgs_data = labeling_data['images']
        self._cats_data = labeling_data['categories']
        self._annotations_data = labeling_data['annotations']
        self.cat_to_label = {i+1: cat['name'] for i, cat in enumerate(self._cats_data)}
        self.cat_to_label[0] = 'background'
        self.n_classes = len(self.cat_to_label)

    def _get_sample_generator(self) -> Generator[Sample]:
        while True:
            random.shuffle(self.samples)
            for i, sample in enumerate(self.samples):
                yield sample

    @property
    def modeltype(self):
        return self._modeltype.replace('_', ' ').capitalize()

    @modeltype.setter
    def modeltype(self, modeltype):
        if not self._is_modeltype_set:
            assert modeltype in ['object_detection', 'image_segmentation']
            self._modeltype = modeltype
            self._is_modeltype_set = True
        else:
            print("Can\'t change modeltype")

    def __repr__(self) -> str:
        return f"{self.modeltype} dataset: \n" \
               f"\t- {len(self.samples)} samples, \n" \
               f"\t- {self.n_classes} categories --> {self.cat_to_label}"

    def __len__(self) -> int:
        return len(self.samples)
