from pathlib import Path
import platform
import lightning as L
import torch
from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
    """LightningDataModule for Fashion MNIST with rotation augmentation and parallel data loading.

    Args:
        data_dir: Directory to store the dataset.
        batch_size: Batch size for training/validation/test.
        num_workers: Number of worker processes for data loading. Set > 0 for parallel loading.
        seed: Random seed for reproducibility.
        subset_fraction: Fraction of dataset to use (for debugging/testing).
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory.
            Recommended when using GPU for faster data transfer.
        persistent_workers: If True, the data loader will not shutdown the worker processes
            after a dataset has been consumed once. Useful when num_workers > 0.
        prefetch_factor: Number of batches loaded in advance by each worker.
            Only used when num_workers > 0.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        subset_fraction: float = 1.0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Define the transform to rotate images by 45 degrees
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation((45, 45)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.batch_size_per_device = batch_size

        # Set multiprocessing context based on platform:
        # - macOS (Darwin): use "fork" for compatibility with M1/M2 chips
        # - Windows: only "spawn" is supported
        # - Linux: None (uses default, which is "fork")
        system = platform.system()
        if system == "Darwin":
            self._multiprocessing_context = "fork"
        elif system == "Windows":
            self._multiprocessing_context = "spawn"
        else:
            self._multiprocessing_context = None

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

        if stage == "fit" or stage is None:
            dataset_full = FashionMNIST(root=self.hparams.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset_full, [50000, 10000], generator=torch.Generator().manual_seed(42)
            )

            # Apply subset if specified
            if self.hparams.subset_fraction < 1.0:
                train_size = int(len(self.train_dataset) * self.hparams.subset_fraction)
                val_size = int(len(self.val_dataset) * self.hparams.subset_fraction)
                self.train_dataset = torch.utils.data.Subset(
                    self.train_dataset,
                    torch.randperm(len(self.train_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[
                        :train_size
                    ],
                )
                self.val_dataset = torch.utils.data.Subset(
                    self.val_dataset,
                    torch.randperm(len(self.val_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[
                        :val_size
                    ],
                )

        if stage == "test":
            self.test_dataset = FashionMNIST(root=self.hparams.data_dir, train=False, transform=self.transform)

            # Apply subset if specified
            if self.hparams.subset_fraction < 1.0:
                test_size = int(len(self.test_dataset) * self.hparams.subset_fraction)
                self.test_dataset = torch.utils.data.Subset(
                    self.test_dataset,
                    torch.randperm(len(self.test_dataset), generator=torch.Generator().manual_seed(self.hparams.seed))[
                        :test_size
                    ],
                )

        if stage == "predict":
            self.predict_dataset = FashionMNIST(root=self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )


if __name__ == "__main__":
    # typer.run(preprocess)
    # dataset = Planetoid(root="data", name="Cora")
    ds = FashionMNIST(root="data", download=True, transform=torchvision.transforms.ToTensor())
    print(len(ds.classes))
