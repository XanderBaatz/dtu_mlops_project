
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

class NN(L.LightningModule):
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size: int = 32,
        output_size: int = 10,
        lr: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

class CNN(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        lr: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28×28 → 14×14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


class C8SteerableCNN(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.r2_act = gspaces.rot2dOnR2(N=8)

        self.input_type = enn.FieldType(
            self.r2_act,
            input_channels * [self.r2_act.trivial_repr],
        )

        out_type1 = enn.FieldType(
            self.r2_act,
            4 * [self.r2_act.regular_repr],
        )

        out_type2 = enn.FieldType(
            self.r2_act,
            2 * [self.r2_act.regular_repr],
        )

        self.features = enn.SequentialModule(
            enn.R2Conv(self.input_type, out_type1, kernel_size=3, padding=1),
            enn.ReLU(out_type1),
            enn.R2Conv(out_type1, out_type2, kernel_size=3, padding=1),
            enn.ReLU(out_type2),
        )

        self.gpool = enn.GroupPooling(out_type2)

        c = self.gpool.out_type.size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.features(x)
        x = self.gpool(x)
        return self.classifier(x.tensor)

    def _step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

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
