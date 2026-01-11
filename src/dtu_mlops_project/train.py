from dtu_mlops_project.model import GCN, C8SteerableCNN
#from dtu_mlops_project.data import pcam_dataset
from torch_geometric.datasets import Planetoid
import torch
import torch_geometric
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer
from typing import Annotated
plt.style.use("ggplot")

app = typer.Typer()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

@app.command()
def train(
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        output: Annotated[str, typer.Option("--output", "-o")] = "model.pth"
    ) -> None:
    print("Training started...")
    print("Device:", DEVICE)
    print(f"{lr=}, {batch_size=}, {epochs=}")

    #model = C8SteerableCNN(n_classes=2).to(DEVICE)
    model = GCN().to(DEVICE)
    dataset = Planetoid(root="data", name="Cora")

    train_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collect statistics
    statistics: dict = {
        "train_loss": [],
        "train_accuracy": [],
    }

    for epoch in range(epochs):
        model.train() # set model to training mode

        for i, data in enumerate(train_dataloader):
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

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, iter {i}, loss: {loss.item()}")

    #for epoch in range(epochs):
    #    model.train()
    #    for i, (img, target) in enumerate(train_dataloader):
    #        img, target = img.to(DEVICE), target.to(DEVICE)
    #        optimizer.zero_grad()
    #        y_pred = model(img)
    #        loss = loss_fn(y_pred, target)
    #        loss.backward()
    #        optimizer.step()
    #        statistics["train_loss"].append(loss.item())
    #
    #        accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
    #        statistics["train_accuracy"].append(accuracy)
    #
    #        if i % 100 == 0:
    #            print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training finished.")
    torch.save(model.state_dict(), f"models/{output}")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.pdf")

if __name__ == "__main__":
    app()
