import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dtu_mlops_project.data import RotatedFashionMNIST, MyDataset


class TestRotatedFashionMNIST:
    """Test for RotatedFashionMNIST data module."""

    @pytest.fixture
    def data_module(self, tmp_path):
        """Create a RotatedFashionMNIST instance with temporary data directory."""
        return RotatedFashionMNIST(
            data_dir=str(tmp_path), batch_size=32, train_val_split=(55_000, 5_000), num_workers=0, subset_fraction=1.0
        )

    def test_initialization(self, data_module):
        """Test that RotatedFashionMNIST initializes correctly."""
        assert data_module.hparams.batch_size == 32
        assert data_module.hparams.num_workers == 0
        assert data_module.hparams.subset_fraction == 1.0

    def test_num_classes(self, data_module):
        """Test that num_classes returns 10 for FashionMNIST."""
        assert data_module.num_classes == 10

    def test_prepare_data(self, data_module):
        """Test that prepare_data downloads the dataset."""
        data_module.prepare_data()
        # Check that data directory exists
        data_path = Path(data_module.hparams.data_dir) / "FashionMNIST" / "raw"
        assert data_path.exists()

    def test_setup_fit_stage(self, data_module):
        """Test setup method with fit stage."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        assert hasattr(data_module, "data_train")
        assert hasattr(data_module, "data_val")
        assert len(data_module.data_train) == 55000
        assert len(data_module.data_val) == 5000

    def test_setup_test_stage(self, data_module):
        """Test setup method with test stage."""
        data_module.prepare_data()
        data_module.setup(stage="test")

        assert hasattr(data_module, "data_test")
        assert len(data_module.data_test) == 10000

    def test_setup_predict_stage(self, data_module):
        """Test setup method with predict stage."""
        data_module.prepare_data()
        data_module.setup(stage="predict")

        assert hasattr(data_module, "data_predict")

    def test_train_dataloader(self, data_module):
        """Test train_dataloader returns a valid DataLoader."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 32

    def test_val_dataloader(self, data_module):
        """Test val_dataloader returns a valid DataLoader."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 32

    def test_test_dataloader(self, data_module):
        """Test test_dataloader returns a valid DataLoader."""
        data_module.prepare_data()
        data_module.setup(stage="test")

        test_loader = data_module.test_dataloader()
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 32

    def test_subset_fraction(self, tmp_path):
        """Test that subset_fraction correctly reduces dataset size."""
        data_module = RotatedFashionMNIST(
            data_dir=str(tmp_path), batch_size=32, train_val_split=(55_000, 5_000), num_workers=0, subset_fraction=0.1
        )
        data_module.prepare_data()
        data_module.setup(stage="fit")

        expected_train_size = int(55000 * 0.1)
        expected_val_size = int(5000 * 0.1)

        assert len(data_module.data_train) == expected_train_size
        assert len(data_module.data_val) == expected_val_size

    def test_batch_loading(self, data_module):
        """Test that batches can be loaded from dataloaders."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        assert len(batch) == 2
        images, labels = batch
        assert images.shape[0] <= 32  # Batch size
        assert images.shape[1] == 1  # Grayscale
        assert images.shape[2] == 28  # Height
        assert images.shape[3] == 28  # Width
        assert labels.shape[0] == images.shape[0]

    def test_transform_applied(self, data_module):
        """Test that transform is applied to images."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        train_loader = data_module.train_dataloader()
        images, _ = next(iter(train_loader))

        # Check that images are tensors with values in [0, 1]
        assert isinstance(images, torch.Tensor)
        assert images.min() >= 0
        assert images.max() <= 1

    def test_different_batch_sizes(self, tmp_path):
        """Test data module with different batch sizes."""
        batch_sizes = [16, 32, 64]

        for batch_size in batch_sizes:
            data_module = RotatedFashionMNIST(data_dir=str(tmp_path), batch_size=batch_size, num_workers=0)
            data_module.prepare_data()
            data_module.setup(stage="fit")

            train_loader = data_module.train_dataloader()
            batch_images, _ = next(iter(train_loader))
            assert batch_images.shape[0] <= batch_size

    def test_train_val_test_split_consistency(self, tmp_path):
        """Test that train/val/test splits are created consistently."""

        data_module = RotatedFashionMNIST(
            data_dir=str(tmp_path), batch_size=32, train_val_split=(55_000, 5_000), num_workers=0, subset_fraction=1.0
        )
        data_module.prepare_data()
        data_module.setup(stage="fit")

        # Verify split sizes are correct
        assert len(data_module.data_train) == 55000
        assert len(data_module.data_val) == 5000

        # All samples should be unique between train and val
        train_indices = set(data_module.data_train.indices)
        val_indices = set(data_module.data_val.indices)
        assert len(train_indices & val_indices) == 0  # No overlap


class TestMyDataset:
    """Test for MyDataset class."""

    @pytest.fixture
    def data_path(self, tmp_path):
        """Create a temporary data path."""
        return tmp_path

    def test_initialization(self, data_path):
        """Test that MyDataset initializes correctly."""
        dataset = MyDataset(data_path=data_path)
        assert dataset.data_path == data_path

    def test_len_method_exists(self, data_path):
        """Test that __len__ method exists."""
        dataset = MyDataset(data_path=data_path)
        assert hasattr(dataset, "__len__")
        assert callable(getattr(dataset, "__len__"))

    def test_getitem_method_exists(self, data_path):
        """Test that __getitem__ method exists."""
        dataset = MyDataset(data_path=data_path)
        assert hasattr(dataset, "__getitem__")
        assert callable(getattr(dataset, "__getitem__"))

    def test_preprocess_method_exists(self, data_path):
        """Test that preprocess method exists."""
        dataset = MyDataset(data_path=data_path)
        assert hasattr(dataset, "preprocess")
        assert callable(getattr(dataset, "preprocess"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
