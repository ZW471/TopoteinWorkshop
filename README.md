# TopoteinWorkshop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

To benchmark the GTNN models introduced in our Topotein paper, we developed TopoteinWorkshop. This is a topological deep learning extension to the [ProteinWorkshop](https://github.com/a-r-j/ProteinWorkshop) framework for protein structure representation learning. This library substantially extends the original benchmark library to incorporate topological features and models for protein structure representation learning.

## Overview

TopoteinWorkshop builds upon ProteinWorkshop to provide:
- Protein Combinatorial Complex data structure
- Novel Geometric Topological Neural Network architectures for protein structure learning, including TCPNet, GNN-TNN, and ETNN
- Benchmarking GTNN models, and evaluating them against existing GGNN models

## Installation

### Environment Setup

```bash
# Create a conda environment using the provided environment file
uv sync
```

We use Python 3.10.13 and primarily run on Cambridge CSD3 Ampere GPU cluster. In most cases, you will need a GPU with high memory like A100 80G for training the larger models. 

### Dependencies

The main dependencies are managed through the conda environment file. Key dependencies include:
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- ProteinWorkshop
- Hydra

## Datasets

For dataset downloads, please refer to the [ProteinWorkshop documentation](https://www.proteins.sh). Topotein uses the same dataset formats and processing pipelines as ProteinWorkshop.

## Usage

### Running a Model

We use Hydra to handle configurations. Check `ProteinWorkshop/proteinworkshop/config` for full details of possible configurations.

Example command to run our model:

```bash
python ./ProteinWorkshop/proteinworkshop/train.py \
  encoder=tcpnet_v0 \
  task=multiclass_graph_classification \
  dataset=fold_superfamily \
  features=ca_bb_sse_3di \
  +aux_task=nn_sequence_3di \
  dataset.datamodule.dataset_fraction=1 \
  logger=wandb \
  trainer.max_epochs=150
```

## Models

Topotein implements several topological neural network architectures:
- TCPNet (`tcpnet_v1` for using protein-level message passing, `tcpnet_v0` for not using this channel)
- GVP-TNN (`tvp`)
- ETNN (`etnn` for our implementation optimized for protein combinatorial complex, `etnn_original` for the original implementation)

## Project Structure

The main content of our framework are stored in the `topotein` folder within the ProteinWorkshop directory. This includes:
- GTNN models in `topotein/models/`
- Protein combinatorial complex and topological featurisation in `topotein/features/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds upon [ProteinWorkshop](https://github.com/a-r-j/ProteinWorkshop)
- We thank the Cambridge CSD3 for computational resources
