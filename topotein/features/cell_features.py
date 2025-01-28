"""Utilities for computing cell features."""
from typing import List, Union

import numpy as np
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
import toponetx as tnx
from proteinworkshop.features.utils import _normalize

CELL_FEATURES: List[str] = [
    "cell_size",
    "node_features",
    "edge_features",
    "cell_type",
]
"""List of cell features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_cell_features(
        x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Computes scalar cell features from a :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` object.

    :param x: :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param features: List of cell features to compute.
    :type features: Union[List[str], ListConfig]

    """
    feats = []
    for feature in features:
        if feature == "cell_size":  # TODO
            feats.append(compute_cell_sizes(x.cells))
        elif feature == "node_features":  # TODO
            n1, n2 = x.x[x.cell_index[0]], x.x[x.cell_index[1]]
            feats.append(torch.cat([n1, n2], dim=1))
        elif feature == "edge_features":
            pass # TODO
        elif feature == "cell_type":
            feats.append(x.cell_type.T)
        elif feature == "orientation":
            raise NotImplementedError
        elif feature == "pos_emb":
            feats.append(pos_emb(x.cell_index))
        else:
            raise ValueError(f"Unknown cell feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1)


@jaxtyped(typechecker=typechecker)
def compute_vector_cell_features(
        x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> Union[Data, Batch]:
    vector_cell_features = []
    for feature in features:
        if feature == "cell_vectors":  # TODO
            E_vectors = x.pos[x.cell_index[0]] - x.pos[x.cell_index[1]]
            vector_cell_features.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.cell_vector_attr = torch.cat(vector_cell_features, dim=0)
    return x


@jaxtyped(typechecker=typechecker)
def compute_cell_sizes(cell_complex: tnx.CellComplex):
    return torch.Tensor(list(map(cell_complex.size, iter(cell_complex.cells))))


@jaxtyped(typechecker=typechecker)
def pos_emb(cell_index: EdgeTensor, num_pos_emb: int = 16):
    raise NotImplementedError
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = cell_index[0] - cell_index[1]

    frequency = torch.exp(
        torch.arange(
            0, num_pos_emb, 2, dtype=torch.float32, device=cell_index.device
        )
        * -(np.log(10000.0) / num_pos_emb)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)
