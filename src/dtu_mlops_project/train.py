from dtu_mlops_project.model import GCN, C8SteerableCNN
#from dtu_mlops_project.data import pcam_dataset
from torch_geometric.datasets import Planetoid
import torch
import torch_geometric
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer
from typing import Annotated
import wandb
import os

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
        epochs: int = 20,
        output: Annotated[str, typer.Option("--output", "-o")] = "model.pth"
    ) -> None:
    print("Training started...")
    print("Device:", DEVICE)
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # W&B: login from .env
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Initialize W&B run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "my_project"),
        entity=os.getenv("WANDB_ENTITY"),
        config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs},
        name=f"GCN_training_{os.getpid()}",
        reinit=True
    )

    # Model and dataset
    #model = C8SteerableCNN(n_classes=2).to(DEVICE)
    model = GCN().to(DEVICE)
    dataset = Planetoid(root="data", name="Cora")
    train_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collect statistics
    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()

        for i, data in enumerate(train_dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Metrics
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            statistics["train_loss"].append(loss.item())
            statistics["train_accuracy"].append(acc)

            # Log to W&B
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": acc,
                "epoch": epoch
            })

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, iter {i}, loss: {loss.item()}")

    print("Training finished.")

    # Save model
    model_path = f"models/{output}"
    torch.save(model.state_dict(), model_path)

    # Log model as W&B artifact
    artifact = wandb.Artifact(
        name=f"GCN_model_{os.getpid()}",
        type="model",
        description="Trained GCN on Cora dataset"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    artifact.wait()  # ensure upload completes

    # Plot statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.pdf")

    # Finish W&B run
    run.finish()


if __name__ == "__main__":
    app()
