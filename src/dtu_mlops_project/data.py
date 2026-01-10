from pathlib import Path

import typer
from torch.utils.data import Dataset

# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/reddit.html#Reddit
from torch_geometric.datasets import Planetoid
#from torch_geometric.data import Dataset, download_url

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


def preprocess(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    data_raw = "data"
    dataset = Planetoid(
        root=data_raw,
        name="Cora",
    )
    #dataset = MyDataset(data_path)
    dataset.process()


if __name__ == "__main__":
    #typer.run(preprocess)
    dataset = Planetoid(root="data", name="Cora")

    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of nodes: {dataset[0].num_nodes}")
    print(f"Number of edges: {dataset[0].num_edges}")
    print(f"Node features: {dataset[0].num_node_features}")
    print(f"Classes: {dataset.num_classes}")
