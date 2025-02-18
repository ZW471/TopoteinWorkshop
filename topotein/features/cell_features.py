"""Utilities for computing cell features."""
from typing import List, Union

import numpy as np
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
import toponetx as tnx
from proteinworkshop.features.utils import _normalize
from torch.nn.utils.rnn import pad_sequence
from proteinworkshop.models.utils import localize, centralize

CELL_FEATURES: List[str] = [
    "cell_size",
    "node_features",
    "edge_features",
    "cell_type",
    "sse_vector_norms",
    "sse_one_hot",
    "sse_variance_wrt_localized_frame"
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
        elif feature == "pos_emb":  # TODO: investigate whether edge pos_emb is actually used first
            feats.append(pos_emb(x.cell_index))
        elif feature == "sse_vector_norms":
            vectors = vector_features(x)
            feats.append(torch.norm(torch.stack(vectors, dim=2), dim=1))
        elif feature == "sse_variance_wrt_localized_frame":
            feats.append(variance_wrt_localized_frame(x))
        else:
            raise ValueError(f"Unknown cell feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1).float()


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
        if feature == "sse_vectors":  # TODO
            sse_vectors = vector_features(x)
            for sse_vector in sse_vectors:
                vector_cell_features.append(_normalize(sse_vector).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    return torch.cat(vector_cell_features, dim=1).float()


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


@jaxtyped(typechecker=typechecker)
def variance_wrt_localized_frame(batch: ProteinBatch) -> torch.Tensor:
    _, X_c = centralize(batch, key='pos', batch_index=batch.batch)
    frames = localize(X_c, batch.sse_cell_index_simple)

    results = []
    n_segments = batch.sse_cell_index_simple.shape[1]
    for i in range(n_segments):
        start = batch.sse_cell_index_simple[0, i]
        end = batch.sse_cell_index_simple[1, i]
        # Process the segment
        seg_result = (X_c[start:end, :] @ frames[i].T).var(dim=0)
        results.append(seg_result)
    # Stack the variance results; shape: (n_segments, output_features)
    return torch.stack(results, dim=0)


@jaxtyped(typechecker=typechecker)
def vector_features(batch: ProteinBatch) -> list[torch.Tensor]:
    """
    Extracts a list of geometric vector features from 3D spatial coordinates and structured
    secondary element (SSE) index data.

    This function computes various vectors derived from the coordinates of the start, end, and
    median positions of SSEs, as well as their relationships to the center of mass (COM) and
    other intermediate positions. The features include vectors such as start-to-end, median-
    to-start, etc. These vectors can be used as input for downstream processing or modeling.

    :param X: A tensor containing the 3D spatial coordinates of residues.
    :param sse_idx: A tensor containing the indices of the start and end residues for
        SSEs, with shape (2, n) where n is the number of SSEs.
    :return: A list of geometric vector features, where each feature is a tensor.
    """

    device = batch.pos.device
    node_idx_in_sse = torch.cat([torch.arange(s, e, device=device) for s, e in batch.sse_cell_index_simple.T], dim=-1)
    sse_batch = torch.cat([torch.ones(e - s, device=device) * idx for idx, (s, e) in enumerate(batch.sse_cell_index_simple.T)], dim=-1).long()
    batch['pos_in_sse'] = batch.pos[node_idx_in_sse]
    com_pos, _ = centralize(batch, key='pos_in_sse', batch_index=sse_batch)
    X = batch.pos

    start_residue_idx = batch.sse_cell_index_simple[0, :]
    end_residue_idx = batch.sse_cell_index_simple[1, :]

    start_residue_pos = X[start_residue_idx, :]
    end_residue_pos = X[end_residue_idx, :]
    middle_pos = (start_residue_pos + end_residue_pos) / 2

    median_residue_idx = (start_residue_idx + end_residue_idx) / 2
    s_median_residue_idx = torch.ceil(median_residue_idx).long()
    e_median_residue_idx = torch.floor(median_residue_idx).long()
    median_residue_pos = ((X[e_median_residue_idx, :] + X[s_median_residue_idx, :]) / 2)

    s_to_e_vec = end_residue_pos - start_residue_pos
    med_to_s_vec = start_residue_pos - median_residue_pos
    med_to_e_vec = end_residue_pos - median_residue_pos
    com_to_s_vec = start_residue_pos - com_pos
    com_to_e_vec = end_residue_pos - com_pos
    med_to_com_vec = com_pos - median_residue_pos
    med_to_middle_vec = middle_pos - median_residue_pos
    middle_to_com_vec = com_pos - middle_pos

    features = [
        s_to_e_vec,
        med_to_s_vec,
        med_to_e_vec,
        com_to_s_vec,
        com_to_e_vec,
        med_to_com_vec,
        med_to_middle_vec,
        middle_to_com_vec
    ]

    return features
