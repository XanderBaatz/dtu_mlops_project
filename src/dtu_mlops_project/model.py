from typing import Any, Dict, Tuple

import torch
from torch import nn, optim
import lightning as L
from lightning import LightningModule

from escnn import gspaces
from escnn import nn as enn

import hydra
from omegaconf import DictConfig


# -------------------------
# Simple Feedforward NN
# -------------------------
class NN(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# -------------------------
# Simple CNN
# -------------------------
class CNN(LightningModule):
    def __init__(
        self,
        optimizer: Dict[str, Any] | None = None,
        net: Dict[str, Any] | None = None,
        num_classes: int = 10,
    ):
        super().__init__()

        # net is intentionally ignored (kept for Hydra compatibility)

        if isinstance(optimizer, DictConfig):
            optimizer = dict(optimizer)

        if optimizer is None:
            optimizer = {
                "_target_": "torch.optim.Adam",
                "_partial_": True,
                "lr": 1e-3,
            }

        self.save_hyperparameters(logger=False)

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        cfg = self.hparams.optimizer
        if cfg.get("_partial_", False):
            opt_fn = hydra.utils.instantiate(cfg)
            return opt_fn(self.parameters())
        return hydra.utils.instantiate(cfg, params=self.parameters())

# -------------------------
# VERY SMALL C8 STEERABLE CNN
# -------------------------
class C8SteerableCNN(LightningModule):
    def __init__(
        self,
        optimizer: Dict[str, Any] | None = None,
        net: Dict[str, Any] | None = None,
    ):
        super().__init__()

        # Convert DictConfig to dict if necessary
        if isinstance(optimizer, DictConfig):
            optimizer = dict(optimizer)
        if isinstance(net, DictConfig):
            net = dict(net)

        # Default net config if not provided
        if net is None:
            net = {
                "input_channels": 1,
                "num_classes": 10,
            }

        # Default optimizer config if not provided
        if optimizer is None:
            optimizer = {
                "_target_": "torch.optim.Adam",
                "_partial_": True,
                "lr": 1e-3,
            }

        self.save_hyperparameters(logger=False)

        # Extract net parameters
        input_channels = net.get("input_channels", 1)
        num_classes = net.get("num_classes", 10)

        self.r2_act = gspaces.rot2dOnR2(N=8)
        self.input_type = enn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

        out_type = enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr])

        self.block = enn.SequentialModule(
            enn.R2Conv(self.input_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.ReLU(out_type, inplace=True),
            enn.PointwiseMaxPool(out_type, 2),
            enn.GroupPooling(out_type),
        )

        # After MaxPool2d(2): 28x28 -> 14x14, and GroupPooling reduces 4*8 -> 4 channels
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block(x)
        x = self.classifier(x.tensor.flatten(1))
        return x

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        cfg = self.hparams.optimizer
        if cfg.get("_partial_", False):
            opt_fn = hydra.utils.instantiate(cfg)
            return opt_fn(self.parameters())
        return hydra.utils.instantiate(cfg, params=self.parameters())
