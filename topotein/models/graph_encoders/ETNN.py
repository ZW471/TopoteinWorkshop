import hydra
import torch
from beartype.typing import Set
from omegaconf import DictConfig
from toponetx import CombinatorialComplex, CellComplex

from proteinworkshop import constants
from topomodelx.base.message_passing import MessagePassing
from topomodelx.base.aggregation import Aggregation
from topomodelx.utils.sparse import from_sparse
from torch_scatter import scatter_add

from topotein.models.graph_encoders.layers.ETNN import ETNNLayer


class ETNNModel(torch.nn.Module):
    def __init__(self, in_dim0=57, in_dim1=2, in_dim2=4, emb_dim=512, dropout=0.1, num_layers=6, activation="relu"):
        super(ETNNModel, self).__init__()
        self.layers = torch.nn.ModuleList([ETNNLayer(emb_dim, in_dim1, in_dim2, dropout, activation) for _ in range(num_layers)])
        self.emb_0 = torch.nn.Linear(in_dim0, emb_dim)

    def forward(self, batch):
        X = batch.pos
        H0 = self.emb_0(batch.x)
        H1 = batch.edge_attr
        H2 = batch.sse_attr

        device = X.device

        # ccc is taking way more time when compared to cell complex
        # ccc: CombinatorialComplex = batch.sse_cell_complex.to_combinatorial_complex()
        #
        # N2_0 = from_sparse(ccc.incidence_matrix(rank=0, to_rank=2).T)
        # N1_0 = from_sparse(ccc.incidence_matrix(rank=0, to_rank=1).T)
        # N0_0_via_1 = from_sparse(ccc.adjacency_matrix(rank=0, via_rank=1))
        # N0_0_via_2 = from_sparse(ccc.adjacency_matrix(rank=0, via_rank=2))
        cc: CellComplex = batch.sse_cell_complex
        Bt = [from_sparse(cc.incidence_matrix(rank=i, signed=False).T).to(device) for i in range(1,3)]
        N2_0 = (torch.sparse.mm(Bt[1], Bt[0]) / 2).coalesce()
        N1_0 = Bt[0].coalesce()
        N0_0_via_1 = from_sparse(cc.adjacency_matrix(rank=0, signed=False)).to(device)
        N0_0_via_2 = torch.sparse.mm(N2_0.T, N2_0).coalesce()


        for layer in self.layers:
            H0_update, X_update = layer(X, H0, H1, H2, N0_0_via_1, N0_0_via_2, N2_0, N1_0)
            H0 = H0 + H0_update
            X = X + X_update
        return {
            "node_embedding": H0,
            # "edge_embedding": torch.cat([H1, H1], dim=-1),
            # "sse_embedding": H2,
            "pos": X
        }

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_attr", "batch", "N2_0", "N1_0", "N0_0_via_1", "N0_0_via_2", "sse_attr"}


@hydra.main(
    version_base="1.3",
    config_path=str(constants.SRC_PATH / "config" / "encoder"),
    config_name="etnn.yaml",
)
def _main(cfg: DictConfig):
    print(cfg)
    enc = hydra.utils.instantiate(cfg)
    print(enc)


if __name__ == "__main__":
    _main()
