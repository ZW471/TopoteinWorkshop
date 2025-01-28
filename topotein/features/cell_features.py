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
from torch.nn.utils.rnn import pad_sequence

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
        if feature == "sse_size":
            feats.append(compute_cell_sizes(x.sse_cell_complex).to(x.x.device))
        elif feature == "node_features":
            feats.append(get_means_by_group(x.x, x.sse_cell_index))
        elif feature == "edge_features":  # TODO: but unsure if this is needed because edge features contains too much node features
            raise NotImplementedError
        elif feature == "sse_one_hot":
            feats.append(x.sse.to(x.x.device))
        elif feature == "orientation":
            raise NotImplementedError
        elif feature == "pos_emb":  # TODO: investigate whether edge pos_emb is actually used first
            feats.append(pos_emb(x.cell_index))
        else:
            raise ValueError(f"Unknown cell feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1)


@jaxtyped(typechecker=typechecker)
def compute_vector_cell_features(
        x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Compute vector-based cell features for a given data or batch object.

    This function processes a data or batch object to generate vector-based
    features for cell-centric data representation. It computes vector
    attributes for cells based on positional information provided in the input
    data or batch object. The resulting features are then stored in the input
    object as an attribute. The operation supports only specific known
    features.

    :param x: A data structure, either of type ``Data`` or ``Batch``, which
              contains cell-related positional information as attributes such
              as ``pos`` and ``cell_index``.
    :param features: A list (or ``ListConfig``) of feature names to be
                     generated and added to the ``Data`` or ``Batch`` object.
    :return: The modified data structure of type ``Data`` or ``Batch`` with
             updated cell vector attributes.
    :raises ValueError: If an unknown vector feature is specified in the
                        ``features`` parameter.
    """
    vector_cell_features = []
    for feature in features:
        if feature == "cell_vectors":  # TODO
            E_vectors = x.pos[x.cell_index[0]] - x.pos[x.cell_index[1]]
            vector_cell_features.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    return torch.cat(vector_cell_features, dim=0)


@jaxtyped(typechecker=typechecker)
def compute_cell_sizes(cell_complex: tnx.CellComplex) -> torch.Tensor:
    """
    Compute the sizes of all cells within the provided cell complex.

    This function iterates over all the cells in the given cell complex
    and computes their sizes. The returned tensor contains the sizes of
    each cell, in order of their iteration.

    :param cell_complex: The cell complex whose cell sizes need to be computed.
    :type cell_complex: tnx.CellComplex

    :return: A tensor containing the sizes of all cells in the given
        cell complex.
    :rtype: torch.Tensor
    """
    return torch.tensor(list(map(cell_complex.size, iter(cell_complex.cells))))


@jaxtyped(typechecker=typechecker)
def get_means_by_group(features: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """
    Calculate the mean of feature vectors for each specified group in an efficient manner.

    The method processes a tensor of feature vectors and a list of group indices.
    Each group's feature vectors are extracted and padded to handle variable group sizes.
    The mean is then calculated across feature vectors in each group.

    :param features: A tensor containing feature vectors of shape (N, M), where N is
        the number of samples and M is the number of features.
    :type features: torch.Tensor
    :param groups: A list of tuples containing indices specifying the groups. Each tuple
        corresponds to a group, and the indices within the tuple represent the positions
        of rows in the `features` tensor that belong to that group.
    :type groups: list[tuple[int]]
    :return: A tensor of means for each group of shape (J, M), where J is the number of
        groups and M matches the feature size from the input.
    :rtype: torch.Tensor
    """
    # Step 1: Create a list of groups of rows from x.x using indexing
    groups = [features[indices, :] for indices in groups]

    # Handle variable-size groups with padding
    groups_padded = pad_sequence(groups, batch_first=True)  # Shape (J, max_rows, 60)

    # Step 2: Calculate the mean for each group efficiently
    means = groups_padded.mean(dim=1)  # Shape (J, 60)

    return means


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
