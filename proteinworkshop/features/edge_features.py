"""Utilities for computing edge features."""
from typing import List, Union

import numpy as np
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch import nn
from torch_geometric.data import Batch, Data

from proteinworkshop.features.utils import _normalize
from proteinworkshop.models.graph_encoders.components import radial

EDGE_FEATURES: List[str] = [
    "edge_distance",
    "node_features",
    "edge_type",
    "sequence_distance",
]
"""List of edge features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Computes scalar edge features from a :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` object.

    :param x: :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param features: List of edge features to compute.
    :type features: Union[List[str], ListConfig]

    """
    feats = []
    for feature in features:
        if feature == "edge_distance":
            feats.append(compute_edge_distance(x.pos, x.edge_index))
        elif feature == "node_features":
            n1, n2 = x.x[x.edge_index[0]], x.x[x.edge_index[1]]
            feats.append(torch.cat([n1, n2], dim=1))
        elif feature == "edge_type":
            feats.append(x.edge_type.T)
        elif feature == "orientation":
            raise NotImplementedError
        elif feature == "sequence_distance":
            feats.append(x.edge_index[1] - x.edge_index[0])
        elif feature == "rbf":
            distance = compute_edge_distance(x.pos, x.edge_index)
            feats.append(radial.compute_rbf(distance))
        elif feature == "rbf_16":
            distance = compute_edge_distance(x.pos, x.edge_index)
            feats.append(radial.compute_rbf(distance, num_rbf=16, max_distance=20.0))
        elif feature == "pos_emb":
            feats.append(pos_emb(x.edge_index))
        elif feature == "dist_pos_emb":
            distance = compute_edge_distance(x.pos, x.edge_index)
            pos_enc = DistancePositionalEncoding(num_frequencies=8,
                                                 max_frequency=5.0,
                                                 include_original=False).to(x.pos.device)
            feats.append(pos_enc(distance))
        else:
            raise ValueError(f"Unknown edge feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1)


@jaxtyped(typechecker=typechecker)
def compute_vector_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> Union[Data, Batch]:
    vector_edge_features = []
    for feature in features:
        if feature == "edge_vectors":
            E_vectors = x.pos[x.edge_index[0]] - x.pos[x.edge_index[1]]
            vector_edge_features.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.edge_vector_attr = torch.cat(vector_edge_features, dim=0)
    return x

class DistancePositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 16,
                 max_frequency: float = 100.0,
                 include_original: bool = False):
        """
        Args:
          num_frequencies: how many different ω_k to use
          max_frequency: the largest ω_k (you can also space them logarithmically)
          include_original: if True, appends the raw d itself to the encoding
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # here we space frequencies linearly from 1.0 up to max_frequency
        self.register_buffer('freq_bands',
                             torch.linspace(1.0, max_frequency, num_frequencies))

        self.include_original = include_original

    def forward(self, d: torch.Tensor):
        """
        d: tensor of shape (E,) or (B, E) containing distances
        returns: tensor of shape (…, D) where D = num_frequencies*2 (+1 if include_original)
        """
        # ensure shape (..., 1)
        orig_shape = d.shape
        d = d.unsqueeze(-1)  # now (..., 1)

        # compute angles: (..., num_frequencies)
        # you can also multiply by 2*pi here if you like
        angles = d * self.freq_bands  # broadcast: (…, 1) * (num_frequencies,) → (…, num_frequencies)

        # stack sin and cos: → (…, num_frequencies*2)
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)

        if self.include_original:
            pe = torch.cat([d, pe], dim=-1)

        return pe


@jaxtyped(typechecker=typechecker)
def compute_edge_distance(
    pos: CoordTensor, edge_index: EdgeTensor
) -> torch.Tensor:
    """
    Compute the euclidean distance between each pair of nodes connected by an edge.

    :param pos: Tensor of shape :math:`(|V|, 3)` containing the node coordinates.
    :type pos: CoordTensor
    :param edge_index: Tensor of shape :math:`(2, |E|)` containing the indices of the nodes forming the edges.
    :type edge_index: EdgeTensor
    :return: Tensor of shape :math:`(|E|, 1)` containing the euclidean distance between each pair of nodes connected by an edge.
    :rtype: torch.Tensor
    """
    return torch.pairwise_distance(
        pos[edge_index[0, :]], pos[edge_index[1, :]]
    )


@jaxtyped(typechecker=typechecker)
def pos_emb(edge_index: EdgeTensor, num_pos_emb: int = 16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(
            0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device
        )
        * -(np.log(10000.0) / num_pos_emb)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)
