import functools
from typing import List, Literal, Optional, Tuple, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

import toponetx as tnx

from topotein.features.sse import get_sse_cell_group


@typechecker
def compute_cells(
        x: Union[Data, Batch, Protein, ProteinBatch],
        cell_types: Union[ListConfig, List[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orchestrates the computation of cells for a given data object.

    This function returns a tuple of tensors, where the first tensor is a
    tensor indicating the cell type of shape (``|E|``) and the second are the
    cell indices of shape (``2 x |E|``).

    The cell type tensor can be used to mask out cells of a particular type
    downstream.

    :param x: The input data object to compute cells for
    :type x: Union[Data, Batch, Protein, ProteinBatch]
    :param cell_types: List of cell types to compute. Must be a sequence of
        ``knn_{x}``, ``eps_{x}``, (where ``{x}`` should be replaced by a
        numerical value) ``seq_forward``, ``seq_backward``.
    :type cell_types: Union[ListConfig, List[str]]
    :raises ValueError: Raised if ``x`` is not a ``torch_geometric`` Data or
        Batch object
    :raises NotImplementedError: Raised if an cell type is not implemented
    :return: Tuple of tensors, where the first tensor is a tensor indicating
        the cell type of shape (``|E|``) and the second are the cell indices of
        shape (``2 x |E|``).
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    # Handle batch
    cc = to_cell_complex(x)  # TODO: attach cells

    # Iterate over cell types
    for cell_type in cell_types:
        if cell_type.startswith("sse"):
            sse_requirements = list(map(int, cell_type.split("_")[1:]))
            cc.complex['sse_type_num'] = sse_requirements[0]
            if len(sse_requirements) == 2:
                sse_minimal_size = sse_requirements[1]
            else:
                sse_minimal_size = 3
            cells = get_sse_cell_group(x.sse, cc.complex['sse_type_num'], minimal_group_size=sse_minimal_size)
            for key, val in cells.items():
                cc.add_cells_from(val, rank=2, sse_type=key, cell_type=cell_type)

        else:
            raise NotImplementedError(f"Cell type {cell_type} not implemented")

    # Compute cell types
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(cells)
        ],
        dim=0,
    ).unsqueeze(0)
    cells = torch.cat(cells, dim=1)

    return cells, indxs

def to_cell_complex(x: Union[Data, Batch, Protein, ProteinBatch]) -> tnx.CellComplex:
    cc = tnx.CellComplex()
    cc._add_nodes_from(list(range(x.num_nodes)))
    print(cc.number_of_nodes())

    for edge_relation in range(x.num_relation):
        cc.add_edges_from(
            ebunch_to_add=x.edge_index.T[(x.edge_type == edge_relation).squeeze()].detach().cpu().numpy(),
            edge_type=edge_relation,
        )
    return cc