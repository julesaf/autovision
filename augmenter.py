from __future__ import annotations
import albumentations as A
import random
import copy
import tqdm
from .sample import Sample
from .bbox import Bbox


class Augmenter:
    def __init__(self, samples: list, pipeline=None) -> None:
        self.samples = samples
        self.pipeline = pipeline
        if pipeline is None:
            self.pipeline = A.Compose([
                A.Rotate(p=0.6, value=[220, 220, 220], border_mode=0, limit=8),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0.75,
                label_fields=['class_labels']
            ))

    def run(self, n_augmented: int) -> list[Sample]:
        augmented_samples = []
        for i in tqdm.auto.trange(n_augmented):
            random_idx = random.randrange(len(self.samples))
            random_sample = copy.deepcopy(self.samples[random_idx])
            random_sample.mode = 1
            augmented_sample = self.pipeline(
                image=random_sample.get_img(img_format='channel_last'),
                bboxes=[bbox.coord for bbox in random_sample.bboxes],
                class_labels=[bbox.label for bbox in random_sample.bboxes]
            )
            augmented_sample = Sample(
                _id=str(i),
                img=augmented_sample['image'],
                bboxes=[
                    Bbox(_id=str(i), coord=coord, label=augmented_sample['class_labels'][i])
                    for i, coord in enumerate(augmented_sample['bboxes'])
                ]
            )
            augmented_samples.append(augmented_sample)
        return augmented_samples
