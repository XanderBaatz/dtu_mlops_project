from torch_geometric.nn import GCNConv
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


# Define a simple CNN model
class CNN(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = optim.Adam,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # CNN model
        self.model = nn.Sequential(
            nn.Conv2d(self.hparams.net.input_channels, 8, self.hparams.net.kernel_size, padding=self.hparams.net.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, self.hparams.net.kernel_size, padding=self.hparams.net.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Neural network classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, self.hparams.net.num_classes)
        )

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


# Define a simple two-layer GCN
class GCN(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        features: int = 1433,
        num_classes: int = 7,
    ):
        super().__init__()

        self.lr = lr

        self.save_hyperparameters(logger=False)

        self.conv1 = GCNConv(features, 32)  # Input: 1433 features, Output: 32 features
        self.conv2 = GCNConv(32, num_classes)   # Output: 7 classes for classification

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        # Apply the first convolution and activation
        x = self.conv1(x, edge_index).relu()
        # Apply the second convolution
        x = self.conv2(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterion(y_pred[data.train_mask], data.y[data.train_mask])
        acc = (y_pred[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #self.logger.experiment.log({"logits": wandb.Histogram(y_pred.cpu().detach().numpy())})
        return loss

    def test_step(self, batch, batch_idx) -> None:
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterion(y_pred[data.test_mask], data.y[data.test_mask])
        acc = (y_pred[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()

        # Logging
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterion(y_pred[data.val_mask], data.y[data.val_mask])
        acc = (y_pred[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).float().mean()

        # Logging
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# Define steerable CNN model
class C8SteerableCNNOld(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Symmetry group C8
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # Input type (scalar field)
        self.input_type = enn.FieldType(
            self.r2_act,
            input_channels * [self.r2_act.trivial_repr]
        )

        # Steerable CNN layers
        self.features = enn.SequentialModule(
            enn.R2Conv(
                self.input_type,
                enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            enn.InnerBatchNorm(
                enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
            ),
            enn.ReLU(
                enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
                inplace=True,
            ),

            enn.R2Conv(
                enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
                enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr]),
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            enn.InnerBatchNorm(
                enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr])
            ),
            enn.ReLU(
                enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr]),
                inplace=True,
            ),

            enn.R2Conv(
                enn.FieldType(self.r2_act, 4 * [self.r2_act.regular_repr]),
                enn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr]),
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            enn.InnerBatchNorm(
                enn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr])
            ),
            enn.ReLU(
                enn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr]),
                inplace=True,
            ),
        )

        # Group pooling
        self.gpool = enn.GroupPooling(
            enn.FieldType(self.r2_act, 2 * [self.r2_act.regular_repr])
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.features(x)
        x = self.gpool(x)
        x = self.classifier(x.tensor)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# Define steerable CNN model
class C8SteerableCNN(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Equivariance under rotations by 45 degrees
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # Input type (scalar field)
        self.input_type = enn.FieldType(
            self.r2_act,
            input_channels * [self.r2_act.trivial_repr]
        )

        # Convolution 1
        out_type = enn.FieldType(
            self.r2_act,
            24 * [self.r2_act.regular_repr]
        )
        self.block1 = enn.SequentialModule(
            enn.MaskModule(self.input_type, 28, margin=1),
            enn.R2Conv(
                self.input_type,
                out_type,
                kernel_size=7,
                padding=1,
                bias=False,
            ),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )

        # Convolution 2
        in_type = self.block1.out_type
        out_type = enn.FieldType(
            self.r2_act,
            48 * [self.r2_act.regular_repr]
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )
        self.pool1 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # Convolution 3
        in_type = self.block2.out_type
        out_type = enn.FieldType(
            self.r2_act,
            48 * [self.r2_act.regular_repr]
        )
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )

        # Convolution 4
        in_type = self.block3.out_type
        out_type = enn.FieldType(
            self.r2_act,
            96 * [self.r2_act.regular_repr]
        )
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )
        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # Convolution 5
        in_type = self.block4.out_type
        out_type = enn.FieldType(
            self.r2_act,
            96 * [self.r2_act.regular_repr]
        )
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )

        # Convolution 6
        in_type = self.block5.out_type
        out_type = enn.FieldType(
            self.r2_act,
            64 * [self.r2_act.regular_repr]
        )
        self.block6 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )

        # Adaptive pooling to 1x1 before group pooling
        self.adaptive_pool = enn.PointwiseAdaptiveAvgPool(out_type, output_size=1)

        self.gpool = enn.GroupPooling(
            out_type
        )

        # Number of output channels
        c = self.gpool.out_type.size

        # FC
        self.fully_net = nn.Sequential(
            nn.Linear(c, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.Linear(64, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.adaptive_pool(x)
        x = self.gpool(x)
        x = self.fully_net(x.tensor.reshape(x.tensor.shape[0], -1))
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

    print("Graph Convolutional Network:")
    model = GCN()
    print(model)
    print()
