from torch.utils.data import Dataset
from dtu_mlops_project.data import RotatedFashionMNIST
import torch


def test_my_dataset():
    """Test the MyDataset class."""

    # Datamodule
    dm = RotatedFashionMNIST(data_dir="data")
    dm.prepare_data()
    dm.setup(stage='fit')
    dl = dm.train_dataloader()

    # Train dataset
    dataset = dm.train_dataset

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 50000  # Check length of training dataset
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2  # (image, label)
    image, label = sample
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 28, 28)  # Check image shape
    assert isinstance(label, int)
    assert 0 <= label < dm.num_classes  # Check label range


if __name__ == "__main__":
    test_my_dataset()
