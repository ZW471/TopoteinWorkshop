import functools
from beartype.typing import List, Tuple, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

import toponetx as tnx

from topotein.features.sse import get_sse_cell_group

from numpy import unique

CATEGORY_DSSP_3 = {"H", "E", "C"}
CATEGORY_DSSP_8 = {"H", "B", "E", "G", "I", "T", "S", "C"}


@typechecker
def compute_sses(
        x: Union[Data, Batch, Protein, ProteinBatch],
        sse_types: Union[ListConfig, List[str]],
        directed_edges=False
) -> Tuple[torch.Tensor, Tuple[Tuple[int, ...], ...], tnx.CellComplex]:
    """
    Orchestrates the computation of cells for a given data object.

    This function returns a tuple of tensors, where the first tensor is a
    tensor indicating the cell type of shape (``|C|``) and the second are the
    cell indices of shape (``2 x |C|``).

    The cell type tensor can be used to mask out cells of a particular type
    downstream.

    :param x: The input data object to compute cells for
    :type x: Union[Data, Batch, Protein, ProteinBatch]
    :param sse_types: List of cell types to compute. Must be a sequence of
        ``knn_{x}``, ``eps_{x}``, (where ``{x}`` should be replaced by a
        numerical value) ``seq_forward``, ``seq_backward``.
    :type sse_types: Union[ListConfig, List[str]]
    :raises ValueError: Raised if ``x`` is not a ``torch_geometric`` Data or
        Batch object
    :raises NotImplementedError: Raised if a cell type is not implemented
    :return: Tuple of tensors, where the first tensor is a tensor indicating
        the cell type of shape (``|C|``) and the second are the cell indices of
        shape (``2 x |C|``).
    :rtype: Tuple[torch.Tensor, Tuple[Tuple[int, ...], ...], tnx.CellComplex]
    """
    # check if sse types are valid
    sse_types = set(sse_types)
    if len(sse_types.difference(CATEGORY_DSSP_3)) == 0:  # no type that is not in 3-class category
        is_using_simple_categories = True
    elif len(sse_types.union(CATEGORY_DSSP_8)) > 0:  # some types are in the 8-class category
        is_using_simple_categories = False
    else:  # some types are out of the 8-class category
        raise ValueError(f"invalid sse types, valid sse types are: {CATEGORY_DSSP_8}")

    # construct cell complex
    sse_group_cc = to_cell_complex(x, directed=directed_edges)
    sse_group_cc.complex['sse_type'] = list(sse_types)

    if is_using_simple_categories:
        sse_group_cc.complex['sse_type_num'] = len(sse_types)

        cells = get_sse_cell_group(x.sse, x.batch, sse_group_cc.complex['sse_type_num'], minimal_group_size=3)
        for key, val in cells.items():
            sse_group_cc.add_cells_from(val, rank=2, sse_type=key)
    else:
        raise NotImplementedError(f"only 3-class scheme implemented")

    # cell index (the nodes contained in a cell)
    sse_cells = tuple(sse_group_cc.get_cell_attributes('sse_type', rank=2).keys())
    sse_cell_types = list(sse_group_cc.get_cell_attributes('sse_type', rank=2).values())
    _, sse_cell_types = unique(sse_cell_types, return_inverse=True)
    sse_cell_types = torch.tensor(sse_cell_types, dtype=torch.long)
    sse_cell_types = torch.nn.functional.one_hot(sse_cell_types, num_classes=len(sse_types))

    return sse_cell_types, sse_cells, sse_group_cc


@typechecker
def to_cell_complex(x: Union[Data, Batch, Protein, ProteinBatch], directed=False) -> tnx.CellComplex:
    cc = tnx.CellComplex()
    cc._add_nodes_from(list(range(x.num_nodes)))
    if directed:
        cc._G = cc._G.to_directed()
    for edge_relation in range(x.num_relation):
        cc.add_edges_from(
            ebunch_to_add=x.edge_index.T[(x.edge_type == edge_relation).squeeze()].detach().cpu().numpy(),
            edge_type=edge_relation,
        )
    return cc


@typechecker
def get_cell_attr_onehot(cell_complex: tnx.CellComplex, attr_name: str):
    cell_types = torch.tensor(list(cell_complex.get_cell_attributes(attr_name, rank=2).values()), dtype=torch.long)
    class_num = f'{attr_name}_num'
    cell_type_onehot = torch.nn.functional.one_hot(cell_types, num_classes=cell_complex.complex[class_num])
    return cell_type_onehot