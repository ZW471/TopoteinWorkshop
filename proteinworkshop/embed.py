import collections
import json
import sys
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial

import hydra
import lightning as L
import omegaconf
import torch
from beartype.typing import Dict, List, Optional, Union


from loguru import logger as log
from tqdm import tqdm

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel


def process_batch(batch, model, device, split, jsonl_path, use_chromadb=False, collection=None):
    """Process a single batch to generate embeddings"""
    ids = batch.id
    batch = batch.to(device)
    batch = model.featuriser(batch)
    out = model.forward(batch)
    graph_embeddings = out["graph_embedding"]
    node_embeddings = out["node_embedding"]

    if use_chromadb:
        collection.add(embeddings=node_embeddings, ids=ids)
    else:
        # Write embeddings directly to file
        with open(jsonl_path, 'a') as f:
            for idx, id_val in enumerate(ids):
                g_emb = graph_embeddings[idx].detach().cpu().tolist()
                n_emb = node_embeddings[batch.batch == idx].detach().cpu().tolist()
                f.write(json.dumps({
                    "id": id_val,
                    "node_embedding": n_emb,
                    "graph_embedding": g_emb,
                    "split": split
                }) + '\n')

    return len(ids)  # Return number of processed samples for tracking progress


def embed(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "A checkpoint path must be provided."
    if cfg.use_cuda_device and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    L.seed_everything(cfg.seed)

    log.info("Instantiating datamodule:... ")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating model:... ")
    model: L.LightningModule = BenchMarkModel(cfg)

    # Initialize lazy layers for parameter counts
    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup(stage="lazy_init")  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        out = model.forward(batch)
        log.info(f"Model output: {out}")
        del batch, out

    # Load weights
    log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu" if cfg.trainer.accelerator == "cpu" else None)["state_dict"]

    if cfg.finetune.encoder.load_weights:
        encoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder"):
                encoder_weights[k.replace("encoder.", "")] = v
        log.info(f"Loading encoder weights: {encoder_weights}")
        err = model.encoder.load_state_dict(encoder_weights, strict=False)
        log.warning(f"Error loading encoder weights: {err}")

    if cfg.finetune.decoder.load_weights:
        decoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("decoder"):
                decoder_weights[k.replace("decoder.", "")] = v
        log.info(f"Loading decoder weights: {decoder_weights}")
        err = model.decoder.load_state_dict(decoder_weights, strict=False)
        log.warning(f"Error loading decoder weights: {err}")
    else:
        model.decoder = None

    log.info("Freezing encoder!")
    for param in model.encoder.parameters():
        param.requires_grad = False

    log.info("Freezing decoder!")
    model.decoder = None  # TODO make this controllable by config

    # Select CUDA computation device, otherwise default to CPU
    if cfg.use_cuda_device:
        device = torch.device(f"cuda:{cfg.cuda_device_index}")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    # Setup datamodule
    datamodule.setup()

    # Prepare storage for embeddings
    if cfg.use_chromadb:
        log.info("Using ChromaDB for embedding storage")

        import chromadb
        from chromadb.config import Settings
        assert cfg.collection_name, "A collection name must be provided when use_chromadb is True."
        # Initialise chromadb
        chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=".chromadb",  # Optional, defaults to .chromadb/ in the current directory
                anonymized_telemetry=False,
            )
        )
        chroma_client.persist()
        collection = chroma_client.create_collection(name=cfg.collection_name)
    else:
        log.info("Using JSONL file for embedding storage")
        assert cfg.jsonl_filepath, "A JSONL filepath must be provided when use_chromadb is False."
        jsonl_path = Path(os.path.join(os.environ["EMBED_PATH"], cfg.jsonl_filepath, "embeddings.jsonl"))
        # Create parent directories if they don't exist
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # Create new empty file if it exists
        with open(jsonl_path, 'w') as f:
            pass

    # Determine number of processes to use
    num_workers = min(mp.cpu_count() - 1, 12)  # Use up to 4 CPU cores or less if not available
    log.info(f"Using {num_workers} workers for parallel processing")

    # Iterate over batches and perform embedding
    dataloaders = {}
    if "train" in cfg.embed.split:
        dataloaders["train"] = datamodule.train_dataloader()
    if "val" in cfg.embed.split:
        dataloaders["val"] = datamodule.val_dataloader()
    if "test" in cfg.embed.split:
        dataloaders["test"] = datamodule.test_dataloader()

    # Process all splits
    for split, dataloader in dataloaders.items():
        log.info(f"Performing embedding for split: {split}")

        # Process batches in parallel if not using CUDA
        if cfg.use_cuda_device:
            # If using CUDA, process sequentially
            for batch in tqdm(dataloader):
                process_batch(
                    batch=batch,
                    model=model,
                    device=device,
                    split=split,
                    jsonl_path=jsonl_path,
                    use_chromadb=cfg.use_chromadb,
                    collection=collection if cfg.use_chromadb else None
                )
        else:
            # For CPU processing, we can use multiprocessing
            # First, collect all batches
            all_batches = list(dataloader)

            # Create a process pool
            with mp.Pool(num_workers) as pool:
                # Create a partial function with fixed parameters
                process_fn = partial(
                    process_batch,
                    model=model,
                    device=device,
                    split=split,
                    jsonl_path=jsonl_path,
                    use_chromadb=cfg.use_chromadb,
                    collection=collection if cfg.use_chromadb else None
                )

                # Map the function to all batches with a progress bar
                for _ in tqdm(
                        pool.imap_unordered(process_fn, all_batches),
                        total=len(all_batches),
                        desc=f"Processing {split} split"
                ):
                    pass

    # Persist storage for ChromaDB
    if cfg.use_chromadb:
        chroma_client.persist()
        log.info("ChromaDB embeddings persisted successfully")
    else:
        log.info(f"Embeddings have been written to: {jsonl_path}")


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="embed",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    embed(cfg)


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