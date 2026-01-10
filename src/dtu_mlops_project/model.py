from torch import nn
import torch
from torch_geometric.nn import GCNConv

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

# Define a simple two-layer GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 32)  # Input: 1433 features, Output: 32 features
        self.conv2 = GCNConv(32, 7)   # Output: 7 classes for classification

    def forward(self, x, edge_index):
        # Apply the first convolution and activation
        x = self.conv1(x, edge_index).relu()
        # Apply the second convolution
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    model = GCN()
    x = torch.rand(1, 1433)  # Batch size of 1, 1433 features
    edge_index = torch.tensor([[0], [0]])  # Dummy edge index
    print(f"Output shape of model: {model(x, edge_index).shape}")
