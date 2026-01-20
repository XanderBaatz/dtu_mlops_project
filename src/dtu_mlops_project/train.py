from typing import Any, Dict, List
from loguru import logger

import os
from pathlib import Path

import hydra
import lightning as L
import torch


from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from dotenv import load_dotenv

from lightning.pytorch.profilers import PyTorchProfiler

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

# Resolve configs directory absolutely so Hydra works both locally and in containers
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")

# Debug: print resolved config path REMOVE IN PUSH!!!
import sys
if "--config-name" in sys.argv or "--cfg" in sys.argv:
    logger.info(f"Resolved CONFIG_PATH: {CONFIG_PATH}")
    logger.info(f"Config path exists: {Path(CONFIG_PATH).exists()}")
    if Path(CONFIG_PATH).exists():
        logger.info(f"Config contents: {list(Path(CONFIG_PATH).iterdir())[:10]}")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# WandB configuration
api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train")
def train(cfg: DictConfig) -> Dict[str, Any] | None:
    # Setup profiler
    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{cfg.paths.root_dir}/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Seed
    if cfg.get("seed"):
        L.seed_everything(seed=cfg.seed)

    # Instantiate datamodule
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Callbacks
    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = []
    for _, cb_conf in cfg.get("callbacks", {}).items():
        callbacks.append(hydra.utils.instantiate(cb_conf))

    # Logger
    logger.info("Instantiating loggers...")
    loggers: List[Logger] = []
    for _, lg_conf in cfg.get("logger", {}).items():
        loggers.append(hydra.utils.instantiate(lg_conf))

    # Trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers, profiler=profiler)

    # Training
    if cfg.get("train", True):
        logger.info("Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Testing
    if cfg.get("test", True):
        logger.info("Starting testing...")
        test_metrics: Dict[str, Any] = {}
        # ckpt_path = None
        # for cb in callbacks:
        #    if isinstance(cb, ModelCheckpoint):
        #        ckpt_path = cb.best_model_path or None
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(model=model, datamodule=datamodule)
        test_metrics = trainer.callback_metrics

        return test_metrics

    return None


if __name__ == "__main__":
    train()
