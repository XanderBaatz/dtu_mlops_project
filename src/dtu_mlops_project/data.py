from pathlib import Path
import platform
import lightning as L
import torch
import torchvision
from typing import Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt

plt.style.use("ggplot")


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


class RandomRotate:
    def __init__(self, n: int):
        assert n > 0, "n must be positive"
        self.n = n
        self.step = 360 / n

    def __call__(self, img):
        k = torch.randint(0, self.n, (1,), generator=torch.Generator().manual_seed(42)).item()
        angle = k * self.step
        return torchvision.transforms.functional.rotate(img, angle)


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
        train_val_split: Tuple[int, int] = (55_000, 5_000),
        num_workers: int = 0,
        pin_memory: bool = False,
        subset_fraction: float = 1.0,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Define the transform to rotate images by 45 degrees
        self.transform = torchvision.transforms.Compose(
            [
                RandomRotate(8),
                torchvision.transforms.ToTensor(),
            ]
        )

        # Initialize datasets as None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        # Dataset properties
        self._dims = (1, 28, 28)
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

    @property
    def dims(self) -> tuple:
        """Get the shape of a single sample (channels, height, width).

        :return: Shape tuple for FashionMNIST images (1, 28, 28).
        """
        return self._dims

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

        # Load and split data only if not already loaded
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = FashionMNIST(root=self.hparams.data_dir, train=True, transform=self.transform)
            self.data_test = self.data_predict = FashionMNIST(
                root=self.hparams.data_dir, train=False, transform=self.transform
            )
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train.name = "RotatedFashionMNIST Train"
            self.data_val.name = "RotatedFashionMNIST Val"
            self.data_test.name = "RotatedFashionMNIST Test"
            self.data_predict.name = "RotatedFashionMNIST Predict"

        # Subset datasets if specified
        if self.hparams.subset_fraction < 1.0:

            def subset_dataset(dataset):
                subset_size = int(len(dataset) * self.hparams.subset_fraction)
                indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:subset_size]
                return torch.utils.data.Subset(dataset, indices)

            self.data_train = subset_dataset(self.data_train)
            self.data_val = subset_dataset(self.data_val)
            self.data_test = subset_dataset(self.data_test)
            self.data_predict = subset_dataset(self.data_predict)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            multiprocessing_context=self._multiprocessing_context if self.hparams.num_workers > 0 else None,
        )


def dataset_statistics(data_dir: str = "data") -> None:
    dataset = RotatedFashionMNIST(data_dir=data_dir)
    dataset.prepare_data()
    dataset.setup()
    train_dataset = dataset.data_train
    test_dataset = dataset.data_test

    # Basic statistics
    print("Training Set Statistics:")
    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print("Test Set Statistics:")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # First 25 samples
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(25):
        img, label = train_dataset[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.savefig("reports/figures/sample_images.png")
    plt.close()

    # Label distribution
    train_label_distribution = torch.bincount(train_dataset.dataset.targets)
    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.pdf")
    plt.close()

    test_label_distribution = torch.bincount(test_dataset.targets)
    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test Set Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.pdf")
    plt.close()


if __name__ == "__main__":
    # ds = FashionMNIST(root="data", download=True, transform=torchvision.transforms.ToTensor())
    # print(len(ds.classes))
    dataset_statistics(data_dir="data")
