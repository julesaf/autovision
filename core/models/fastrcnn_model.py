from __future__ import annotations

import torch
import numpy as np
import transformers
import torchvision
import pytorch_lightning as pl
import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from .model import Model
from core.batch import BatchWrapper, Batch
from core.sample.sample import Sample
from core.dataset import Dataset


class FastRCNNModel(Model):
    def __init__(self, train_dataset: Dataset = None, valid_dataset: Dataset = None, model_ckpt_filepath: str = None,
                 cat_to_label: dict = None, use_cpu_only=False) -> None:
        super().__init__(train_dataset, valid_dataset, model_ckpt_filepath, cat_to_label)
        #weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        #model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0)
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, box_score_thresh=0.0
        )

        if model_ckpt_filepath is not None:
            model_ckpt = torch.load(model_ckpt_filepath)
            self._configure_model(model_ckpt['n_out_features'])
            self.model.load_state_dict(model_ckpt['model_state_dict'])
            if model_ckpt.get('cat_to_label') is not None:
                self.cat_to_label = model_ckpt['cat_to_label']
        elif train_dataset is not None:
            self._configure_model(train_dataset.n_classes)


        if use_cpu_only:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
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

    def train(self, n_epochs=1, batch_size=32, evaluate_train_set=True, size=None) -> None:
        n_steps_per_epoch = int(len(self.train_dataset) / batch_size)
        if len(self.train_dataset) < batch_size:
            n_steps_per_epoch = 1
            batch_size = len(self.train_dataset)
        n_training_steps = int(n_epochs * len(self.train_dataset) / batch_size)
        self.configure_optimizers(n_training_steps)
        assert(n_epochs >= 1)
        assert (n_steps_per_epoch >= 1)
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch + 1}/{n_epochs}:')
            loss_epoch = []
            for _ in tqdm.trange(n_steps_per_epoch):
                batch = self.train_dataset.get_batch(batch_size=batch_size, size=size, box_format=1)
                loss, lr_state = self._train_step(batch)
                loss_epoch.append(loss)
            print(f'\t--> lr: {lr_state[0]:.6f} - loss: {np.mean(loss_epoch):.3f}')
            if evaluate_train_set:
                self.evaluate(self.train_dataset, batch_size, size)
            if self.valid_dataset is not None:
                self.evaluate(self.valid_dataset, batch_size, size)

    def _train_step(self, batch: Batch):
        self.model.train()
        self.optimizer.zero_grad()
        self._preprocess_data(batch)
        X = torch.Tensor(batch.pixel_values).type(torch.float).to(self.device)
        y = [{
            'boxes': torch.Tensor(batch.bboxes[i]).type(torch.int64).to(self.device),
            'labels': torch.Tensor(batch.categories[i]).type(torch.int64).to(self.device)
        } for i in range(len(batch))]
        self._check_input_data(X, y)
        outputs = self.model(X, y)
        loss = sum(loss for loss in outputs.values())
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return float(loss.item()), self.lr_scheduler.get_last_lr()

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_in_features': self.n_in_features,
            'n_out_features': self.n_out_features,
            'cat_to_label': self.cat_to_label
        }, path)

    def evaluate(self, dataset: Dataset, batch_size=32, size=None) -> None:
        metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
        n_steps_per_epoch = int(len(dataset) / batch_size)
        if len(dataset) < batch_size:
            n_steps_per_epoch = 1
            batch_size = len(dataset)
        self.model.eval()
        for _ in tqdm.trange(n_steps_per_epoch):
            batch = dataset.get_batch(batch_size=batch_size, size=size, box_format=1)
            self._preprocess_data(batch)
            X = torch.Tensor(batch.pixel_values).type(torch.float).to(self.device)
            with torch.no_grad():
                y_pred = self.model(X)
            y_true = [{
                'boxes': torch.Tensor(batch.bboxes[i]).type(torch.int64).to(self.device),
                'labels': torch.Tensor(batch.categories[i]).type(torch.int64).to(self.device)
            } for i in range(len(batch))]
            metric.update(y_pred, y_true)
        for metric, value in metric.compute().items():
            print(f"{metric} = {value.numpy()}")

    def predict(self, samples: list[Sample], size: tuple[int, int] = None) -> list:
        predictions = []
        self.model.eval()
        for sample in tqdm.auto.tqdm(samples):
            batch = BatchWrapper(samples=[sample]).get(size=size, box_format=1)
            self._preprocess_data(batch)
            X = torch.Tensor(batch.pixel_values).type(torch.float).to(self.device)
            self._check_input_data(X)
            with torch.no_grad():
                prediction = self.model(X)[0]
                prediction['labels'] = np.array(
                    [self.cat_to_label.get(int(cat_pred)) for cat_pred in prediction['labels']]
                )
                prediction['boxes'] = prediction['boxes'].cpu()
                prediction['scores'] = prediction['scores'].cpu()
            predictions.append(prediction)
            torch.cuda.empty_cache()
        return predictions

    def _configure_model(self, n_out_features: int):
        self.n_in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.n_out_features = n_out_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            self.n_in_features, self.n_out_features
        )

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
