from typing import Set

import torch
from omegaconf import DictConfig
from topomodelx.utils.sparse import from_sparse
from toponetx import CellComplex
from torch import nn

from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput

import hydra

from proteinworkshop import constants


class TopoteinModelV0(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            node_input_dim: int = 60,  # TODO: use model_cfg as used in the GCPNet model
            edge_input_dim: int = 122,
            cell_input_dim: int = 64,
            node_emb_dim: int = 128,
            edge_emb_dim: int = 64,
            cell_emb_dim: int = 128,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.cell_input_dim = cell_input_dim

        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim
        self.cell_emb_dim = cell_emb_dim

        self.sig = torch.nn.functional.sigmoid
        self.bns = [
            torch.nn.BatchNorm1d(node_emb_dim),
            torch.nn.BatchNorm1d(edge_emb_dim),
            torch.nn.BatchNorm1d(cell_emb_dim),
        ]
        self.relu = torch.nn.ReLU()

        input_dims = [
            self.node_input_dim,
            self.edge_input_dim,
            self.cell_input_dim,
        ]

        emb_dims = [
            self.node_emb_dim,
            self.edge_emb_dim,
            self.cell_emb_dim,
        ]

        self.fc_input = [torch.nn.Linear(input_dim, emb_dim) for input_dim, emb_dim in zip(input_dims, emb_dims)]

        self.fc_emb = [
            [torch.nn.Linear(from_dim, to_dim) for to_dim in emb_dims] for from_dim in emb_dims
        ]

        self.pool = get_aggregation("mean")

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_index", "batch"}

    def forward(self, batch) -> EncoderOutput:

        X = [
            self.fc_input[0](batch.x),
            self.fc_input[1](batch.edge_attr),
            self.fc_input[2](batch.sse_attr)
        ]

        cc:  CellComplex = batch.sse_cell_complex
        B = [from_sparse(cc.incidence_matrix(rank=r, signed=False)) for r in range(1, 3)]
        A = [from_sparse(cc.adjacency_matrix(rank=r, signed=False)) for r in range(2)]
        coA_1 = from_sparse(cc.coadjacency_matrix(rank=1, signed=False))

        # M2_0 = torch.mm(B[1].T, B[0].T) / 2
        # M2_1 = torch.mm(M2_0, B[0])
        # M2_2 = torch.mm(torch.mm(M2_0, A[0]), M2_0.T)
        #
        # M1_2 = B[1]
        # M1_1 = A[1] + coA_1
        # M1_0 = B[0].T
        #
        # M0_2 = M2_0.T
        # M0_1 = B[0]
        # M0_0 = A[0]

        M2_0 = torch.zeros(
            (B[1].shape[-1], B[0].shape[0]),
            dtype=torch.float32,
            device=X[0].device,)
        M2_1 = B[1].T
        M2_2 = torch.zeros(
            (B[1].shape[-1], B[1].shape[-1]),
            dtype=torch.float32,
            device=X[0].device,)

        M1_2 = B[1]
        M1_1 = A[1]
        M1_0 = B[0].T

        M0_2 = M2_0.T
        M0_1 = B[0]
        M0_0 = torch.zeros(
            (A[0].shape[0], A[0].shape[0]),
            dtype=torch.float32,
            device=X[0].device,)

        M = [
            [M0_0, M0_1, M0_2],
            [M1_0, M1_1, M1_2],
            [M2_0, M2_1, M2_2],
        ]

        device = X[0].device
        M = [[element.to(device) for element in row] for row in M]

        H = [None, None, None]

        for _ in range(self.num_layers):
            for to_rank in range(3):
                H[to_rank] = torch.stack([
                    self.fc_emb[from_rank][to_rank](
                        torch.mm(M[from_rank][to_rank].T, X[from_rank])
                    )
                    for from_rank in range(3)]).sum(0)

            for i in range(3):
                X[i] = self.relu(self.bns[i](H[i] + X[i]))

        return EncoderOutput({
            "node_embedding": X[0],
            "edge_embedding": X[1],
            "cell_embedding": X[2],
            "graph_embedding": self.pool(X[0], batch.batch)
        })

@hydra.main(
    version_base="1.3",
    config_path=str(constants.SRC_PATH / "config" / "encoder"),
    config_name="topotein_v0.yaml",
)
def _main(cfg: DictConfig):
    print(cfg)
    enc = hydra.utils.instantiate(cfg)
    print(enc)


if __name__ == "__main__":
    _main()

