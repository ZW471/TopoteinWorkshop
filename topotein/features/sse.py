from typing import Literal
from beartype import beartype as typechecker
import pydssp
from graphein.protein.tensor import Protein
from graphein.protein.tensor.data import ProteinBatch
import torch


@typechecker
def annotate_protein_sses_pydssp(protein: Protein) -> torch.Tensor:
    """Annotate secondary structure elements for a protein using PyTorch. Much faster than P-SEA!"""

    return pydssp.assign(protein.coords[:, :4, :], out_type='onehot').float()

@typechecker
def sse_onehot(protein_batch: ProteinBatch) -> torch.Tensor:
    """Annotate secondary structure elements for a protein using PyTorch. Much faster than P-SEA!"""

    return torch.cat(protein_batch.protein_apply(annotate_protein_sses_pydssp))


