from dtu_mlops_project.model import GCN
#from dtu_mlops_project.data import MyDataset
from torch_geometric.datasets import Planetoid
import torch
import torch_geometric
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use("ggplot")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    print("Training started...")
    print("Device:", DEVICE)
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = GCN().to(DEVICE)
    dataset = Planetoid(root="data", name="Cora")

    train_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collect statistics
    statistics: dict = {
        "train_loss": [],
        "train_accuracy": [],
    }

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train() # set model to training mode

        for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)):
            data = data.to(DEVICE)
            optimizer.zero_grad() # clear previous gradients

            # Forward pass
            out = model(data.x, data.edge_index)

            # Compute loss only for training nodes
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # Log statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            statistics["train_accuracy"].append(accuracy)

    print("Training finished.")
    torch.save(model.state_dict(), "models/model.pth")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.pdf")


if __name__ == "__main__":
    train(epochs=200)
