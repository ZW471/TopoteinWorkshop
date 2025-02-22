import os
import pathlib
import tarfile
from typing import Callable, Dict, Iterable, List, Literal, Optional

import omegaconf
import pandas as pd
import requests
import torch
import urllib3
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger
from loguru import logger as log
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

urllib3.disable_warnings(InsecureRequestWarning)

def flatten_dir(dir: os.PathLike):
    """
    Flattens the nested directory structure of a directory into a single level.
    """
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            try:
                os.rename(
                    os.path.join(dirpath, filename),
                    os.path.join(dir, filename),
                )
            except OSError:
                print(f"Could not move {os.path.join(dirpath, filename)}")


class FoldClassificationDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        split: str,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        in_memory: bool = False,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = pathlib.Path(path)
        if not os.path.exists(self.data_dir):
            log.info(f"Creating dataset directory: {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)

        self.split = split

        self.scop_url = "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-95-1.75.tgz"
        self.structure_dir = self.data_dir / "pdbstyle-1.75"

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.dataset_fraction = dataset_fraction
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle_labels = shuffle_labels

        self.prepare_data_per_node = True

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

    @property
    def test_dataset_names(self) -> List[str]:
        """Provides a list of test set split names."""
        return ["fold", "family", "superfamily"]

    def download(self):
        self.download_data_files()
        self.download_structures()

    def download_data_files(self):  # sourcery skip: move-assign
        """Downloads dataset index files."""
        logger.info(
            f"Downloading Protein Function. Fraction {self.dataset_fraction}"
        )
        BASE_URL = "https://raw.githubusercontent.com/phermosilla/IEConv_proteins/master/Datasets/data/HomologyTAPE/"

        TRAIN_URL = f"{BASE_URL}training.txt"
        VALIDATION_URL = f"{BASE_URL}validation.txt"
        LABELS_URL = f"{BASE_URL}class_map.txt"
        TEST_FAMILY_URL = f"{BASE_URL}test_family.txt"
        TEST_FOLD_URL = f"{BASE_URL}test_fold.txt"
        TEST_SUPERFAMILY_URL = f"{BASE_URL}test_superfamily.txt"

        if not os.path.exists(self.data_dir / "training.txt"):
            logger.info(f"Downloading training data to {self.data_dir}...")
            wget.download(TRAIN_URL, out=str(self.data_dir / "training.txt"))
        if not os.path.exists(self.data_dir / "validation.txt"):
            logger.info(f"Downloading validation data to {self.data_dir}...")
            wget.download(
                VALIDATION_URL, out=str(self.data_dir / "validation.txt")
            )
        if not os.path.exists(self.data_dir / "test_fold.txt"):
            logger.info(f"Downloading test fold data to {self.data_dir}...")
            wget.download(
                TEST_FOLD_URL, out=str(self.data_dir / "test_fold.txt")
            )
        if not os.path.exists(self.data_dir / "test_family.txt"):
            logger.info(f"Downloading test family data to {self.data_dir}...")
            wget.download(
                TEST_FAMILY_URL, out=str(self.data_dir / "test_family.txt")
            )
        if not os.path.exists(self.data_dir / "test_superfamily.txt"):
            logger.info(
                f"Downloading test superfamily data to {self.data_dir}..."
            )
            wget.download(
                TEST_SUPERFAMILY_URL,
                out=str(self.data_dir / "test_superfamily.txt"),
            )
        if not os.path.exists(self.data_dir / "class_map.txt"):
            logger.info(f"Downloading class map data to {self.data_dir}...")
            wget.download(LABELS_URL, out=str(self.data_dir / "class_map.txt"))

    def download_structures(self):  # sourcery skip: extract-method
        """Downloads SCOPe structures with a progress bar."""
        tarfile_path = self.data_dir / "pdbstyle-sel-gs-bib-95-1.75.tgz"

        if not os.path.exists(tarfile_path):
            log.info(f"Downloading SCOPe structures from: {self.scop_url} to {self.data_dir}")
            try:
                # Get metadata about the file (like total size for the progress bar)
                with requests.get(self.scop_url, stream=True, timeout=30) as response:
                    response.raise_for_status()  # Raise for bad HTTP responses
                    total = int(response.headers.get('content-length', 0))  # Total file size in bytes

                    # Display a progress bar with tqdm
                    with open(tarfile_path, "wb") as f, tqdm(
                            desc="Downloading",
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=1024):  # Download file in chunks
                            f.write(chunk)
                            bar.update(len(chunk))  # Update progress bar

            # Fallback: Handle SSL errors
            except requests.exceptions.SSLError as ssl_error:
                log.warning("SSL certificate verification failed. Retrying without certificate validation...")
                log.warning(f"SSL error details: {ssl_error}")

                # Retry download without SSL validation
                with requests.get(self.scop_url, stream=True, timeout=30, verify=False) as response:
                    response.raise_for_status()
                    total = int(response.headers.get('content-length', 0))

                    with open(tarfile_path, "wb") as f, tqdm(
                            desc="Downloading (Insecure)",
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                            bar.update(len(chunk))

            except requests.exceptions.RequestException as e:
                log.error(f"Failed to download the file: {e}")
                raise

            log.info(f"Successfully downloaded to {tarfile_path}")
        else:
            log.info(f"Found SCOP structure tarfile in: {tarfile_path}")

        if not os.path.exists(self.structure_dir):
            log.info(f"Extracting tarfile to {self.data_dir}")
            with tarfile.open(tarfile_path) as tar:
                tar.extractall(str(self.data_dir))
            log.info("Flattening directory")
            flatten_dir(self.structure_dir)
        else:
            log.info("Found SCOPe structures in: ")  # TODO

    def parse_class_map(self) -> Dict[str, str]:
        log.info(f"Reading labels from: {self.data_dir / 'class_map.txt'}")
        class_map = pd.read_csv(
            self.data_dir / "class_map.txt", sep="\t", header=None
        )
        return dict(class_map.values)

    def _get_dataset(self, split: str) -> ProteinDataset:
        if hasattr(self, f"{split}_ds"):
            return getattr(self, f"{split}_ds")

        df = self.parse_dataset(split)
        ds = ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=str(self.structure_dir),
            pdb_codes=list(df.id),
            format="ent",
            graph_labels=[torch.tensor(a) for a in list(df.label)],
            overwrite=self.overwrite,
            transform=self.transform,
            in_memory=self.in_memory,
        )
        setattr(self, f"{split}_ds", ds)
        return ds

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("training")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("validation")

    def test_dataset(
        self, split: Literal["fold", "family", "superfamily"]
    ) -> ProteinDataset:
        return self._get_dataset(f"test_{split}")

    def train_dataloader(self) -> ProteinDataLoader:
        self.train_ds = self.train_dataset()
        return ProteinDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        self.val_ds = self.val_dataset()
        return ProteinDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(
        self, split: Literal["fold", "family", "superfamily"]
    ) -> ProteinDataLoader:
        self.test_ds = self.test_dataset(split)
        return ProteinDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def parse_dataset(self, split: str) -> pd.DataFrame:
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Load ID: label mapping
        class_map = self.parse_class_map()

        # Read in IDs of structures in split
        data = pd.read_csv(
            self.data_dir / f"{split}.txt", sep="\t", header=None
        )

        logger.info(f"Found {len(data)} original examples in {split}")
        # Assign columns to DataFrame
        if len(data.columns) == 4:
            data.columns = ["id", "length", "label", "label_dup"]
        elif len(data.columns) == 3:
            data.columns = ["id", "label", "label_dup"]
        else:
            raise ValueError(
                f"Unexpected number of columns in dataset file ({len(data.columns)})"
            )

        # Map labels to IDs in dataframe
        data["label"] = data.label.map(class_map)
        logger.info(f"Identified {len(data['label'].unique())} classes")

        if self.shuffle_labels:
            logger.info("Shuffling labels. Expecting random performance.")
            data["label"] = data["label"].sample(frac=1).values

        return data

    def parse_labels(self):
        raise NotImplementedError

    def exclude_pdbs(self):
        pass


if __name__ == "__main__":
    import hydra

    from proteinworkshop import constants

    # Fold Dataset
    cfg = omegaconf.OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "dataset" / "fold_fold.yaml"
    )
    cfg.datamodule.path = (
        pathlib.Path(constants.DATA_PATH) / "FoldClassification"
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print(datamodule)
    datamodule.setup()

    # Family Dataset
    cfg = omegaconf.OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "dataset" / "fold_family.yaml"
    )
    cfg.datamodule.path = (
        pathlib.Path(constants.DATA_PATH) / "FoldClassification"
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print(datamodule)
    datamodule.setup()

    # Superfamily Dataset
    cfg = omegaconf.OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "dataset" / "fold_superfamily.yaml"
    )
    cfg.datamodule.path = (
        pathlib.Path(constants.DATA_PATH) / "FoldClassification"
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print(datamodule)
    datamodule.setup()
