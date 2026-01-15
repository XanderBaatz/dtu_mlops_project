from torch_geometric.nn import GCNConv
from torch import nn, optim
from escnn import gspaces
from escnn import nn as enn
import lightning as L
import wandb

# Base model class example (Pytorch Lightning)
# Inspired by https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class Model(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters()

        self.model = ...  # Define your model architecture here

        self.criterium = ... # Define your loss function here

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
        self.save_hyperparameters()

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
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterium(y_pred, targets)
        acc = (y_pred.argmax(1) == targets).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterium(y_pred, targets)
        acc = (y_pred.argmax(1) == targets).float().mean()

        # Logging
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        y_pred = self(inputs)
        loss = self.criterium(y_pred, targets)
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
        input_channels: int = 1,
        kernel_size: int = 3,
        pool_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        num_classes: int = 10,
        lr: float = 1e-2,
    ):
        super().__init__()

        self.lr = lr

        self.save_hyperparameters()

        # CNN model
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size), # [N, 64, 26]
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size), # [N, 32, 24]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size), # [N, 16, 22]
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size),  # [N, 8, 20]
            nn.ReLU(),
        )

        # Neural network classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),         # [N, 8*20*20]
            nn.Linear(8*20*20, 128),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

        # Loss function
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.model(x)
        out = self.classifier(features)
        return out

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


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

        self.save_hyperparameters()

        self.conv1 = GCNConv(features, 32)  # Input: 1433 features, Output: 32 features
        self.conv2 = GCNConv(32, num_classes)   # Output: 7 classes for classification

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        # Apply the first convolution and activation
        x = self.conv1(x, edge_index).relu()
        # Apply the second convolution
        x = self.conv2(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterium(y_pred[data.train_mask], data.y[data.train_mask])
        acc = (y_pred[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #self.logger.experiment.log({"logits": wandb.Histogram(y_pred.cpu().detach().numpy())})
        return loss

    def test_step(self, batch, batch_idx) -> None:
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterium(y_pred[data.test_mask], data.y[data.test_mask])
        acc = (y_pred[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()

        # Logging
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        data = batch
        y_pred = self(data.x, data.edge_index)
        loss = self.criterium(y_pred[data.val_mask], data.y[data.val_mask])
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

        self.lr = lr

        self.save_hyperparameters()

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

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.features(x)
        x = self.gpool(x)
        x = self.classifier(x.tensor)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
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

        self.lr = lr

        self.save_hyperparameters()

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

        self.criterium = nn.CrossEntropyLoss()

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
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(1) == target).float().mean()

        # Logging
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterium(y_pred, target)
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
