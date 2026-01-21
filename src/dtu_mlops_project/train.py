from typing import Any, Dict, List
from loguru import logger

import os

import hydra
import lightning as L
import torch


from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import DictConfig
from dotenv import load_dotenv


import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# WandB configuration
api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> Dict[str, Any] | None:
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

        # Export to ONNX if requested
        if cfg.get("export_onnx", False):
            logger.info("Exporting model to ONNX format...")
            export_to_onnx(model, datamodule, cfg)

        return test_metrics

    return None


# I have added a constrain that needs to be true in the arg "export_onnx=True"
# Can be found in logs/train/runs/TIMESTAMP/model.onnx
def export_to_onnx(model: LightningModule, datamodule: LightningDataModule, cfg: DictConfig) -> None:
    """Export Lightning model to ONNX format.

    Args:
        model: Trained Lightning model
        datamodule: Data module for input shape information
        cfg: Configuration object
    """
    import os

    # Get input shape from datamodule
    if not hasattr(datamodule, "dims"):
        logger.warning("Datamodule does not have 'dims' attribute, skipping ONNX export")
        return

    input_shape = datamodule.dims
    batch_size = 1

    # Create dummy input sample (assumes image data)
    if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
        # input_shape is [channels, height, width]
        dummy_input = torch.randn(batch_size, *input_shape)
    else:
        logger.warning(f"Unsupported input shape: {input_shape}, skipping ONNX export")
        return

    # Determine output path
    output_dir = cfg.paths.output_dir if hasattr(cfg.paths, "output_dir") else "."
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    try:
        logger.info(f"Converting model to ONNX: {onnx_path}")
        model.to_onnx(
            file_path=onnx_path,
            input_sample=dummy_input,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            verbose=False,
        )
        logger.info(f"Model successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {str(e)}")
        raise


if __name__ == "__main__":
    train()
