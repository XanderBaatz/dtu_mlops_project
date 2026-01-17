from typing import Any, Dict, List
import os
import hydra
import torch
import lightning as L

from lightning import Trainer, Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from dotenv import load_dotenv

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

from src.dtu_mlops_project.data import RotatedFashionMNIST

# -------------------------
# Device
# -------------------------
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> Dict[str, Any] | None:

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Data
    datamodule = hydra.utils.instantiate(cfg.data)

    # Model (manual instantiation avoids Hydra recursion bugs)
    target = cfg.model._target_
    kwargs = {k: v for k, v in cfg.model.items() if k != "_target_"}
    module_name, class_name = target.rsplit(".", 1)

    import importlib
    model_cls = getattr(importlib.import_module(module_name), class_name)
    model: LightningModule = model_cls(**kwargs)

    # Callbacks
    callbacks: List[Callback] = [
        hydra.utils.instantiate(cb) for cb in cfg.get("callbacks", {}).values()
    ]

    # Loggers (WandB safe)
    loggers: List[Logger] = [
        hydra.utils.instantiate(lg) for lg in cfg.get("logger", {}).values()
    ]

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        accelerator=DEVICE,
    )

    if cfg.get("train", True):
        trainer.fit(model, datamodule)

    if cfg.get("test", True):
        trainer.test(model, datamodule)
        return trainer.callback_metrics

    return None


if __name__ == "__main__":
    train()
