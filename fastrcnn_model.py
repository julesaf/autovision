from __future__ import annotations
import torch
import numpy as np
import transformers
import torchvision
import tqdm
from .batch import BatchWrapper, Batch
from .sample import Sample
from .dataset import Dataset


class FastRCNNModel:
    def __init__(self, dataset: Dataset) -> None:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   dataset.n_classes)
        self.dataset = dataset
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = None
        self.lr_scheduler = None

        print(f'Using {self.device}...\n')

    def configure_optimizers(self, n_training_steps, lr: float = 0.001) -> None:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = transformers.get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=n_training_steps
        )

    def train(self, n_epochs=1, batch_size=32, size=None) -> None:
        n_steps_per_epoch = int(len(self.dataset) / batch_size)
        n_training_steps = int(n_epochs * len(self.dataset) / batch_size)
        self.configure_optimizers(n_training_steps)
        assert(n_epochs >= 1)
        assert (n_steps_per_epoch >= 1)
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch + 1}/{n_epochs}:')
            loss_epoch = []
            for _ in tqdm.trange(n_steps_per_epoch):
                batch = self.dataset.get_batch(batch_size=batch_size, size=size, mode=1)
                loss, lr_state = self.train_step(batch)
                loss_epoch.append(loss)
            print(f'\t--> lr: {lr_state[0]:.6f} - loss: {np.mean(loss_epoch):.3f}')

    def train_step(self, batch: Batch):
        self._preprocess_data(batch)
        X = torch.Tensor(batch.pixel_values).type(torch.float).to(self.device)
        y = [{
            'boxes': torch.Tensor(batch.bboxes[i]).type(torch.int64).to(self.device),
            'labels': torch.Tensor(batch.categories[i]).type(torch.int64).to(self.device)
        } for i in range(len(batch))]
        self._check_input_data(X, y)
        self.model.train()
        outputs = self.model(X, y)
        loss = sum(loss for loss in outputs.values())
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return loss.item(), self.lr_scheduler.get_last_lr()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def evaluate(self):
        pass

    def predict(self, samples: list[Sample], size: tuple[int, int] = None) -> list:
        predictions = []
        for sample in tqdm.auto.tqdm(samples):
            batch = BatchWrapper(samples=[sample]).get(size=size, mode=1, normalize_imgs=True)
            self._preprocess_data(batch)
            X = torch.Tensor(batch.pixel_values).type(torch.float).to(self.device)
            self._check_input_data(X)
            self.model.eval()
            prediction = self.model(X)
            predictions.append(prediction)
            torch.cuda.empty_cache()
        return predictions

    @staticmethod
    def _preprocess_data(batch: Batch) -> None:
        batch.normalize_imgs()

    @staticmethod
    def _check_input_data(X, y=None) -> None:
        """
        The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image
        , and should be in 0-1 range. Different images can have different sizes.
        During training, the ground-truth boxes in [x1, y1, x2, y2] format is expected,
        with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        assert (X.max() <= 1.)
        assert (X.min() >= 0.)
        # assert (y['boxes'])
