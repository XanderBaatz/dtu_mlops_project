
from torch import nn, optim
from escnn import gspaces
from escnn import nn as enn
from typing import Any, Dict, Tuple
import lightning as L
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch
import wandb

# Base model class example (Pytorch Lightning)
# Inspired by https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class Model(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = ...  # Define your model architecture here

        self.criterion = ... # Define your loss function here

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
        input_size: int = 28*28,
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
            nn.Linear(hidden_size2, output_size)
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


# Define a CNN model
class CNN(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Very simple CNN - only 2 conv layers with small filters
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3),  # [N, 8, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [N, 8, 13, 13]
            nn.Conv2d(8, 16, kernel_size=3),  # [N, 16, 11, 11]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [N, 16, 5, 5]
        )

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [N, 16*5*5 = 400]
            nn.Linear(16*5*5, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

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
        acc = (y_pred.argmax(1) == target).float().mean()

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
        acc = (y_pred.argmax(1) == target).float().mean()

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
        acc = (y_pred.argmax(1) == target).float().mean()

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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}


# Define steerable C8 model
class C8SteerableCNN(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Equivariance under rotations by 45 degrees (C8 group)
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # Input type (scalar field)
        self.input_type = enn.FieldType(
            self.r2_act,
            input_channels * [self.r2_act.trivial_repr]
        )

        # Very simple steerable CNN - only 2 conv layers with small number of features
        # Convolution 1: input -> 4 regular representations
        out_type1 = enn.FieldType(
            self.r2_act,
            4 * [self.r2_act.regular_repr]  # Only 4 features instead of 24
        )
        self.conv1 = enn.SequentialModule(
            enn.R2Conv(
                self.input_type,
                out_type1,
                kernel_size=3,  # Smaller kernel
                padding=1,
                bias=False,
            ),
            enn.ReLU(out_type1, inplace=True),
        )

        # Pooling 1
        self.pool1 = enn.PointwiseMaxPool(out_type1, kernel_size=2, stride=2)

        # Convolution 2: 4 -> 8 regular representations
        out_type2 = enn.FieldType(
            self.r2_act,
            8 * [self.r2_act.regular_repr]  # Only 8 features
        )
        self.conv2 = enn.SequentialModule(
            enn.R2Conv(
                out_type1,
                out_type2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            enn.ReLU(out_type2, inplace=True),
        )

        # Pooling 2
        self.pool2 = enn.PointwiseMaxPool(out_type2, kernel_size=2, stride=2)

        # Group pooling - average over the group
        self.gpool = enn.GroupPooling(out_type2)

        # Number of output channels after group pooling
        c = self.gpool.out_type.size

        # Very simple classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 7 * 7, num_classes)  # Direct mapping to classes
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.gpool(x)
        x = self.classifier(x.tensor)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    print("Feedforward Neural Network:")
    model = NN()
    print(model)
    print()

    print("Convolutional Neural Network:")
    model = CNN()
    print(model)
    print()
