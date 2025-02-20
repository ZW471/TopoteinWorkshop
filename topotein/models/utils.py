from typing import Optional, Tuple, Union

import torch
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, jaxtyped
from torch_geometric.data import Batch

@jaxtyped(typechecker=typechecker)
def centralize(
        pos: torch.Tensor,
        batch_index: torch.Tensor,
        node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor
]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        entities_centroid = torch_scatter.scatter(
            pos[node_mask], batch_index[node_mask], dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        masked_values = torch.ones_like(pos) * torch.inf
        values = pos[node_mask]
        masked_values[node_mask] = (
                values - entities_centroid[batch_index][node_mask]
        )
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = torch_scatter.scatter(
            pos, batch_index, dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]
        entities_centered = pos - entities_centroid[batch_index]

    return entities_centroid, entities_centered


def lift_features_with_padding(features: torch.Tensor, neighborhood: torch.Tensor) -> torch.Tensor:
    """
    Lifts given features with padding based on the provided neighborhood tensor.

    Given an input feature tensor and a neighborhood mapping, the function creates a new
    feature tensor where specific elements are lifted (copied) into a larger tensor
    with padding, according to the neighborhood indices.

    :param features: Tensor containing the feature data, typically in multi-dimensional
        format. It serves as the source from which features will be lifted
        based on the neighborhood mapping.
    :type features: torch.Tensor
    :param neighborhood: Tensor that defines the mapping for lifting the features. It
        contains indices that specify how features from the input tensor will
        be arranged in the output tensor, and its dimensions correspond to
        the mapping logic.
    :type neighborhood: torch.Tensor
    :return: Tensor with lifted features and padding applied, maintaining the necessary
        alignment as defined by the neighborhood tensor.
    :rtype: torch.Tensor
    """
    lifted_size = neighborhood.size()[0]
    lifted_features_values = features[neighborhood.indices()[1, neighborhood.values() == 1]]
    lifted_features = torch.zeros(lifted_size, *features.shape[1:],
                                  device=features.device,
                                  dtype=features.dtype)
    lifted_features[neighborhood.indices()[0]] = lifted_features_values
    return lifted_features