from dtu_mlops_project.model import GCN, Model, C8SteerableCNN, CNN
from dtu_mlops_project.data import RotatedFashionMNIST
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning as L
import torch
import matplotlib.pyplot as plt
import typer
import pandas as pd
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
import os
plt.style.use("ggplot")

# WandB configuration
api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")

app = typer.Typer()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@app.command()
def train(
        lr: Annotated[float, typer.Option("--lr", "-l")] = 1e-3,
        batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 32,
        epochs: Annotated[int, typer.Option("--epochs", "-e")] = 5,
        seed: Annotated[int, typer.Option("--seed", "-s")] = 42,
        output: Annotated[str, typer.Option("--output", "-o")] = "model.pth"
    ) -> None:

    print("Training started...")
    L.seed_everything(seed)

    print("Device:", DEVICE)

    print(f"{lr=}, {batch_size=}, {epochs=}")

    # DataModule
    dm = RotatedFashionMNIST(batch_size=batch_size, seed=seed)
    dm.prepare_data()
    dm.setup(stage='fit')

    # Model
    model = CNN(num_classes=dm.num_classes, lr=lr).to(DEVICE)
    #model = C8SteerableCNN(input_channels=dm.num_channels, num_classes=dm.num_classes, lr=lr).to(DEVICE)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        patience=3,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        monitor="train_loss",
        mode="min",
    )

    # Logger
    logger = CSVLogger("logs/", name="training")

    # Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

    print("Training finished.")
    torch.save(model.state_dict(), f"models/{output}")

    # Load metrics
    metrics_df = pd.read_csv(f"{logger.log_dir}/metrics.csv")

    # Plot metrics
    if "train_loss_epoch" in metrics_df.columns and "train_acc_epoch" in metrics_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(metrics_df["train_loss_epoch"].dropna())
        axs[0].set_title("Train loss")
        axs[1].plot(metrics_df["train_acc_epoch"].dropna())
        axs[1].set_title("Training accuracy")
        fig.savefig("reports/figures/training_statistics.pdf")

    # Plot validation accuracy and loss if available
    if "val_loss_epoch" in metrics_df.columns and "val_acc_epoch" in metrics_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(metrics_df["val_loss_epoch"].dropna())
        axs[0].set_title("Validation loss")
        axs[1].plot(metrics_df["val_acc_epoch"].dropna())
        axs[1].set_title("Validation accuracy")
        fig.savefig("reports/figures/validation_statistics.pdf")


if __name__ == "__main__":
    app()
