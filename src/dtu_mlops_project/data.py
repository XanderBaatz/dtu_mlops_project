from pathlib import Path
import lightning as L
import torch

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
            seed: int = 42
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = 10
        self.num_channels = 1  # Grayscale images

        # Define the transform to rotate images by 45 degrees
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((45, 45)),
            torchvision.transforms.ToTensor(),
        ])

    def prepare_data(self):
        # Download the dataset
        FashionMNIST(root=self.data_dir, train=True, download=True)
        FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage:str=None):
        if stage == 'fit' or stage is None:
            dataset_full = FashionMNIST(root=self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset_full,
                [50000, 10000],
                generator=torch.Generator().manual_seed(self.seed)
            )

        if stage == 'test':
            self.test_dataset = FashionMNIST(root=self.data_dir, train=False, transform=self.transform)

        if stage == 'predict':
            self.predict_dataset = FashionMNIST(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    #typer.run(preprocess)
    #dataset = Planetoid(root="data", name="Cora")
    ds = FashionMNIST(root="data", download=True, transform=torchvision.transforms.ToTensor())
    print(len(ds.classes))
