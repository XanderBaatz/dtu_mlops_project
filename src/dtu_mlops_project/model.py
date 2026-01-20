from torch import nn, optim
from escnn import gspaces
from escnn import nn as enn
from typing import Any, Dict, Tuple
import lightning as L
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch


# Base model class example (Pytorch Lightning)
# Inspired by https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class Model(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = ...  # Define your model architecture here

        self.criterion = ...  # Define your loss function here

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step not implemented.")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step not implemented.")

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer configuration not implemented.")


# Define a simple feedforward neural network
class NN(L.LightningModule):
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size1: int = 128,
        hidden_size2: int = 64,
        output_size: int = 10,
        lr: float = 1e-2,
    ):
        super().__init__()

        self.lr = lr

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterion(y_pred, targets)
        acc = (y_pred.argmax(1) == targets).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterion(y_pred, targets)
        acc = (y_pred.argmax(1) == targets).float().mean()

        # Logging
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterion(y_pred, targets)
        acc = (y_pred.argmax(1) == targets).float().mean()

        # Logging
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# Define a simple CNN model
class CNN(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # CNN model
        self.model = nn.Sequential(
            nn.Conv2d(
                self.hparams.net.input_channels, 8, self.hparams.net.kernel_size, padding=self.hparams.net.padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, self.hparams.net.kernel_size, padding=self.hparams.net.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Neural network classifier
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, self.hparams.net.num_classes))

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        features = self.model(x)
        out = self.classifier(features)
        return out

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(y_pred, target)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(y_pred, target)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(y_pred, target)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


# Define steerable CNN model that mirrors CNN architecture with C8 equivariance
class C8SteerableCNN(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Build C8-equivariant analogue of the plain CNN: two conv blocks with pooling
        self.r2_act = gspaces.rot2dOnR2(N=8)
        self.input_type = enn.FieldType(self.r2_act, self.hparams.net.input_channels * [self.r2_act.trivial_repr])

        # Match parameter count of CNN (~9.1K) using fewer regular representations
        # (regular repr is 8-dim for C8, so fewer channels are needed to match baseline)
        block1_out = enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr])
        block2_out = enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])

        self.block = enn.SequentialModule(
            enn.R2Conv(
                self.input_type,
                block1_out,
                kernel_size=self.hparams.net.kernel_size,
                padding=self.hparams.net.padding,
                bias=True,
            ),
            enn.ReLU(block1_out, inplace=True),
            enn.PointwiseMaxPool(block1_out, 2),
            enn.R2Conv(
                block1_out,
                block2_out,
                kernel_size=self.hparams.net.kernel_size,
                padding=self.hparams.net.padding,
                bias=True,
            ),
            enn.ReLU(block2_out, inplace=True),
            enn.PointwiseMaxPool(block2_out, 2),
            # Group pooling collapses orientation dimension, yielding 16 channels like the plain CNN
            enn.GroupPooling(block2_out),
        )

        # After two 2x2 pools on 28x28 inputs, we have 7x7 spatial features
        # GroupPooling collapses orientation: 8 regular reps → 8 scalar channels
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 7 * 7, self.hparams.net.num_classes))

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block(x)
        x = x.tensor
        out = self.classifier(x)
        return out

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(y_pred, target)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(y_pred, target)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        (y_pred.argmax(1) == target).float().mean()

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(y_pred, target)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    print("Feedforward Neural Network:")
    model = NN()
    print(model)
    print()

    print("Convolutional Neural Network:")
    model = CNN()
    print(model)
    print()

    print("Steerable CNN:")
    model = C8SteerableCNN()
    print(model)
    print()
