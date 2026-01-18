from typing import Any, Dict, List

import os

import hydra
import lightning as L
import torch
import matplotlib.pyplot as plt

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from dotenv import load_dotenv

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

plt.style.use("ggplot")

# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# WandB configuration
api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> Dict[str, Any] | None:

    # Seed
    if cfg.get("seed"):
        L.seed_everything(seed=cfg.seed)

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Callbacks
    callbacks: List[Callback] = []
    for _, cb_conf in cfg.get("callbacks", {}).items():
        callbacks.append(hydra.utils.instantiate(cb_conf))

    # Logger
    logger: List[Logger] = []
    for _, lg_conf in cfg.get("logger", {}).items():
        logger.append(hydra.utils.instantiate(lg_conf))

    # Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Training
    if cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Testing
    test_metrics: Dict[str, Any] = {}
    if cfg.get("test", True):
        #ckpt_path = None
        #for cb in callbacks:
        #    if isinstance(cb, ModelCheckpoint):
        #        ckpt_path = cb.best_model_path or None
        #trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(model=model, datamodule=datamodule)
        test_metrics = trainer.callback_metrics

        return test_metrics

    return None

    # Plot metrics
    if "train_loss_epoch" in metrics_df.columns and "train_acc_epoch" in metrics_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(metrics_df["train_loss_epoch"].dropna())
        axs[0].set_title("Train loss")
        axs[1].plot(metrics_df["train_acc_epoch"].dropna())
        axs[1].set_title("Training accuracy")
        fig.savefig(f"{figures_path}/training_statistics.pdf")

    # Plot validation accuracy and loss if available
    if "val_loss_epoch" in metrics_df.columns and "val_acc_epoch" in metrics_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(metrics_df["val_loss_epoch"].dropna())
        axs[0].set_title("Validation loss")
        axs[1].plot(metrics_df["val_acc_epoch"].dropna())
        axs[1].set_title("Validation accuracy")
        fig.savefig(f"{figures_path}/validation_statistics.pdf")


if __name__ == "__main__":
    train()
