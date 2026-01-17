from pathlib import Path
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import FashionMNIST
import torchvision


class MyDataset(Dataset):
    """My custom dataset (not used yet)."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError

    def preprocess(self, output_folder: Path) -> None:
        raise NotImplementedError


# Lightning DataModule
class RotatedFashionMNIST(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        seed: int = 42,
        subset_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.subset_fraction = subset_fraction

        self.num_classes = 10
        self.num_channels = 1  # grayscale

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((45, 45)),
            torchvision.transforms.ToTensor(),
        ])

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            full_dataset = FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform,
            )

            n_total = len(full_dataset)
            n_subset = int(n_total * self.subset_fraction)

            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(n_total, generator=generator)[:n_subset]

            subset = Subset(full_dataset, indices)

            # 90 / 10 train-val split
            n_train = int(0.9 * len(subset))
            n_val = len(subset) - n_train

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                subset,
                [n_train, n_val],
                generator=generator,
            )

        if stage == "test":
            self.test_dataset = FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
            )

        if stage == "predict":
            self.predict_dataset = FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=5,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=5,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=5,
            persistent_workers=True,
        )


if __name__ == "__main__":
    dm = RotatedFashionMNIST(batch_size=32, subset_fraction=0.05)
    dm.prepare_data()
    dm.setup("fit")
