from typing import Set

import torch
from topomodelx.utils.sparse import from_sparse
from toponetx import CellComplex
from torch import nn

from ProteinWorkshop.proteinworkshop.models.graph_encoders.layers.egnn import EGNNLayer
from ProteinWorkshop.proteinworkshop.models.utils import get_aggregation
from ProteinWorkshop.proteinworkshop.types import EncoderOutput


class TopoteinModelV0(nn.Module):
    def __init__(
            self,
            num_layers: int = 5,
            node_emb_dim: int = 32,
            edge_emb_dim: int = 32,
            cell_emb_dim: int = 32,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim
        self.cell_emb_dim = cell_emb_dim

        self.sig = torch.nn.functional.sigmoid
        self.bn = torch.nn.BatchNorm1d(3)
        self.relu = torch.nn.ReLU()

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_index", "batch"}

    def forward(self, batch) -> EncoderOutput:
        cc:  CellComplex = batch.sse_cell_complex
        B = [from_sparse(cc.incidence_matrix(rank=r, signed=False)) for r in range(1, 3)]
        A = [from_sparse(cc.adjacency_matrix(rank=r, signed=False)) for r in range(2)]
        coA_1 = from_sparse(cc.coadjacency_matrix(rank=1, signed=False))

        M2_0 = torch.mm(B[1].T, B[0].T) / 2
        M2_1 = torch.mm(M2_0, B[0])
        M2_2 = torch.mm(torch.mm(M2_0, A[0]), M2_0.T)

        M1_2 = B[1]
        M1_1 = A[1] + coA_1
        M1_0 = B[0].T

        M0_2 = M2_0.T
        M0_1 = B[0]
        M0_0 = A[0]

        X = [
            batch.x,
            batch.edge_attr,
            batch.sse_attr
        ]

        h = [None, None, None]

        for _ in range(self.num_layers):
            h[0] = torch.mm(M2_0.T, X[2]) + torch.mm(M1_0.T, X[1]) + torch.mm(M0_0.T, X[0])
            h[1] = torch.mm(M2_1.T, X[2]) + torch.mm(M1_1.T, X[1]) + torch.mm(M0_1.T, X[0])
            h[2] = torch.mm(M2_2.T, X[2]) + torch.mm(M1_2.T, X[1]) + torch.mm(M0_2.T, X[0])

        for i in range(3):
            X[i] = self.relu(self.bn(h[i]))

        return EncoderOutput({
            "node_embedding": h[0],
            "edge_embedding": h[1],
            "cell_embedding": h[2],
            "graph_embedding": self.pool(h, batch.batch)
        })