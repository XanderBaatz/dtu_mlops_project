from types import SimpleNamespace
import torch
import typer
from dtu_mlops_project.model import CNN
from dtu_mlops_project.data import RotatedFashionMNIST
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from typing import Annotated
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def evaluate(model_checkpoint: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # DataModule
    dm = RotatedFashionMNIST()
    dm.prepare_data()
    dm.setup(stage="test")

    # mypy fix
    net_config = SimpleNamespace(
        input_channels=1,  # Fashion MNIST has 1 channel
        kernel_size=3,
        padding=1,
        num_classes=dm.num_classes,
    )

    optimizer = torch.optim.Adam

    # Model
    model = CNN(net=net_config, optimizer=optimizer)  # type: ignore[arg-type]
    model.load_state_dict(torch.load(model_checkpoint))

    # Logger
    logger = CSVLogger("logs/", name="testing")

    # Trainer
    trainer = Trainer(
        accelerator="auto",
        logger=logger,
    )

    # Evaluate
    results = trainer.test(model, datamodule=dm)
    print("Test Results:", results)

    # Load metrics
    metrics_df = pd.read_csv(f"{logger.log_dir}/metrics.csv")

    # Plot metrics
    if "test_loss_step" in metrics_df.columns and "test_acc_step" in metrics_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(metrics_df["test_loss_step"].dropna())
        axs[0].set_title("Test loss")
        axs[1].plot(metrics_df["test_acc_step"].dropna())
        axs[1].set_title("Test accuracy")
        fig.savefig("reports/figures/testing_statistics.pdf")


if __name__ == "__main__":
    app()
