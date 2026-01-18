from pathlib import Path
import lightning as L
import torch
from typing import Any, Dict, Optional, Tuple
import typer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/reddit.html#Reddit
from torch_geometric.datasets import KarateClub
from torchvision.datasets import FashionMNIST
import torchvision

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


# For use with rotation equivariance models
# https://lightning.ai/docs/pytorch/latest/data/datamodule.html#what-is-a-datamodule
class RotatedFashionMNIST(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data",
            batch_size: int = 32,
            num_workers: int = 0,
            seed: int = 42,
            subset_fraction: float = 1.0
        ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Define the transform to rotate images by 45 degrees
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((45, 45)),
            torchvision.transforms.ToTensor(),
        ])

        #self.data_train: Optional[Dataset] = None
        #self.data_val: Optional[Dataset] = None
        #self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        # Download the dataset
        FashionMNIST(root=self.hparams.data_dir, train=True, download=True)
        FashionMNIST(root=self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if stage == 'fit' or stage is None:
            dataset_full = FashionMNIST(root=self.hparams.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset_full,
                [50000, 10000],
                generator=torch.Generator().manual_seed(42)
            )

            # Apply subset if specified
            if self.hparams.subset_fraction < 1.0:
                train_size = int(len(self.train_dataset) * self.hparams.subset_fraction)
                val_size = int(len(self.val_dataset) * self.hparams.subset_fraction)
                self.train_dataset = torch.utils.data.Subset(
                    self.train_dataset,
                    torch.randperm(len(self.train_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[:train_size]
                )
                self.val_dataset = torch.utils.data.Subset(
                    self.val_dataset,
                    torch.randperm(len(self.val_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[:val_size]
                )

        if stage == 'test':
            self.test_dataset = FashionMNIST(root=self.hparams.data_dir, train=False, transform=self.transform)

            # Apply subset if specified
            if self.hparams.subset_fraction < 1.0:
                test_size = int(len(self.test_dataset) * self.hparams.subset_fraction)
                self.test_dataset = torch.utils.data.Subset(
                    self.test_dataset,
                    torch.randperm(len(self.test_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[:test_size]
                )

        if stage == 'predict':
            self.predict_dataset = FashionMNIST(root=self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers, shuffle=False)


if __name__ == "__main__":
    #typer.run(preprocess)
    #dataset = Planetoid(root="data", name="Cora")
    ds = FashionMNIST(root="data", download=True, transform=torchvision.transforms.ToTensor())
    print(len(ds.classes))
