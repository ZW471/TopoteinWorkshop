"""
Main module to load and train the model. This should be the program entry
point.
"""
import copy
import sys
from typing import List, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
import pathlib

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
import types
import collections


def _num_training_steps(
        train_dataset, trainer
) -> int:

    from loguru import logger as log
    """
    Returns total training steps inferred from datamodule and devices.

    :param train_dataset: Training dataloader
    :type train_dataset: ProteinDataLoader
    :param trainer: Lightning trainer
    :type trainer: lightning.pytorch.Trainer
    :return: Total number of training steps
    :rtype: int
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataset) * train_dataset.batch_size
    )

    log.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
            train_dataset.batch_size
            * trainer.accumulate_grad_batches
            * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs


def train_model(
        cfg: DictConfig, encoder: Optional[nn.Module] = None
):  # sourcery skip: extract-method

    from loguru import logger as log
    """
    Trains a model from a config.

    If ``encoder`` is provided, it is used instead of the one specified in the
    config.

    1. The datamodule is instantiated from ``cfg.dataset.datamodule``.
    2. The callbacks are instantiated from ``cfg.callbacks``.
    3. The logger is instantiated from ``cfg.logger``.
    4. The trainer is instantiated from ``cfg.trainer``.
    5. (Optional) If the config contains a scheduler, the number of training steps is
         inferred from the datamodule and devices and set in the scheduler.
    6. The model is instantiated from ``cfg.model``.
    7. The datamodule is setup and a dummy forward pass is run to initialise
    lazy layers for accurate parameter counts.
    8. Hyperparameters are logged to wandb if a logger is present.
    9. The model is compiled if ``cfg.compile`` is True.
    10. The model is trained if ``cfg.task_name`` is ``"train"``.
    11. The model is tested if ``cfg.test`` is ``True``.

    :param cfg: DictConfig containing the config for the experiment
    :type cfg: DictConfig
    :param encoder: Optional encoder to use instead of the one specified in
        the config
    :type encoder: Optional[nn.Module]
    """
    # set seed for random number generators in pytorch, numpy and python.random
    import lightning as L
    L.seed_everything(cfg.seed)

    log.info(
        f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}..."
    )
    import lightning as L
    from graphein.ml.datasets.foldcomp_dataset import FoldCompLightningDataModule
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )
    if isinstance(datamodule, FoldCompLightningDataModule):
        log.info(
            f"Setting up fixed dataloaders for {datamodule.__class__.__name__}"
        )
        def setup_fixed(self, stage: Optional[str] = None):
            if stage == "fit" or stage is None:
                log.info("Preprocessing training data")
                self.train_dataset()
                log.info("Preprocessing validation data")
                self.val_dataset()
            elif stage == "test":
                log.info("Preprocessing test data")
                if hasattr(self, "test_dataset_names"):
                    for split in self.test_dataset_names:
                        setattr(self, f"{split}_ds", self.test_dataset(split))
                else:
                    self.test_dataset()
            elif stage == "lazy_init":
                log.info("Preprocessing validation data")
                self.val_dataset()
        datamodule.setup = types.MethodType(
            setup_fixed, datamodule
        )
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.callbacks.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.loggers.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    import lightning as L
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    if cfg.get("scheduler"):
        if (
                cfg.scheduler.scheduler._target_
                == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
                and cfg.scheduler.interval == "step"
        ):
            datamodule.setup()  # type: ignore
            num_steps = _num_training_steps(
                datamodule.train_dataloader(), trainer
            )
            log.info(
                f"Setting number of training steps in scheduler to: {num_steps}"
            )
            cfg.scheduler.scheduler.warmup_epochs = (
                    num_steps / trainer.max_epochs
            )
            cfg.scheduler.scheduler.max_epochs = num_steps
            log.info(cfg.scheduler)

    log.info("Instantiating model...")
    import lightning as L
    from topotein_la.models.vq_base import VQBenchMarkModel
    model: L.LightningModule = VQBenchMarkModel(cfg)

    if encoder is not None:
        log.info(f"Setting user-defined encoder {encoder}...")
        model.encoder = encoder

    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup(stage="lazy_init")  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        log.info(f"batch type: {type(batch)}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        log.info(f"Example labels: {model.get_labels(batch)}")
        # Check batch has required attributes
        for attr in model.encoder.required_batch_attributes:  # type: ignore
            if not hasattr(batch, attr):
                raise AttributeError(
                    f"Batch {batch} does not have required attribute: {attr} ({model.encoder.required_batch_attributes})"
                )
        out = model(batch)
        log.info(f"Model output: {out}")
        del batch, out

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.logging_utils.log_hyperparameters(object_dict)


    if cfg.get("ckpt_path"):
        log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
        if cfg.trainer.accelerator == "cpu":
            log.warning(
                "Loading weights on CPU."
            )
            state_dict = torch.load(cfg.ckpt_path, map_location=cfg.trainer.accelerator)["state_dict"]
        else:
            state_dict = torch.load(cfg.ckpt_path)["state_dict"]

        encoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder"):
                encoder_weights[k.replace("encoder.", "")] = v
        log.info(f"Loading encoder weights: {encoder_weights}")
        err = model.encoder.load_state_dict(encoder_weights, strict=False)
        log.warning(f"Error loading encoder weights: {err}")

    if cfg.get("compile"):
        log.info("Compiling model!")
        import torch_geometric
        model = torch_geometric.compile(model, dynamic=True)


    no_training = cfg.get("no_training", False)
    if cfg.get("task_name") == "train" and not no_training:
        log.info("Starting training!")
        if cfg.get("log_all", False):
            import wandb
            wandb.watch(model, log="all", log_freq=1000, log_graph=True)
        trainer.fit(
            model=model, datamodule=datamodule
        )
        # Log profiler trace
        import lightning as L
        if cfg.get("profiling", False) and isinstance(trainer.profiler, L.pytorch.profilers.PyTorchProfiler):
            import wandb
            profile_art = wandb.Artifact("trace", type="profile")
            for trace in pathlib.Path(trainer.profiler.dirpath).glob("*.pt.trace.json"):
                profile_art.add_file(trace)
            profile_art.save()


    if cfg.get("test"):
        log.info("Starting testing!")
        if hasattr(datamodule, "test_dataset_names"):
            splits = datamodule.test_dataset_names
            wandb_logger = copy.deepcopy(trainer.logger)
            for i, split in enumerate(splits):
                dataloader = datamodule.test_dataloader(split)
                trainer.logger = False
                log.info(f"Testing on {split} ({i+1} / {len(splits)})...")
                results = trainer.test(
                    model=model, dataloaders=dataloader, ckpt_path="best" if not no_training else cfg.get("ckpt_path")
                )[0]
                results = {f"{k}/{split}": v for k, v in results.items()}
                log.info(f"{split}: {results}")
                import wandb
                wandb_logger.log_metrics(results)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best" if not no_training else cfg.get("ckpt_path"))


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="train_vq_vae",
)
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    # Import and initialize modules that might create thread locks here
    # to avoid pickling issues with submitit
    import graphein
    import lightning as L
    import lovely_tensors as lt
    import torch_geometric
    from graphein.protein.tensor.dataloader import ProteinDataLoader
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger
    from graphein.ml.datasets.foldcomp_dataset import FoldCompLightningDataModule
    from proteinworkshop.configs import config
    from topotein_la.models.vq_base import VQBenchMarkModel
    import wandb

    from loguru import logger as log

    graphein.verbose(False)
    lt.monkey_patch()

    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    train_model(cfg)


def _script_main(args: List[str]) -> None:
    """
    Provides an entry point for the script dispatcher.

    Sets the sys.argv to the provided args and calls the main train function.
    """
    sys.argv = args
    register_custom_omegaconf_resolvers()
    _main()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore
