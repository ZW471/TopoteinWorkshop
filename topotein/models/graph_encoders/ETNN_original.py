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

from proteinworkshop.models.utils import get_aggregation
from topotein.models.graph_encoders.sotas.etnn.model import ETNNCore


class ETNNModel(torch.nn.Module):
    def __init__(self, in_dim0=57, in_dim1=2, in_dim2=4, emb_dim=512, dropout=0.1, num_layers=6, activation="relu",pool="mean",
                 # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
                 # They are simply listed here to highlight key available arguments
                 # They will only be used when debug=True
                 **kwargs):
        super().__init__()

        self.debug = kwargs.get("debug", False)

        if self.debug:
            self.in_dim0 = in_dim0
            self.in_dim1 = in_dim1
            self.in_dim2 = in_dim2
            self.emb_dim = emb_dim
            self.dropout = dropout
            self.num_layers = num_layers
            self.activation = activation
            self.pool = pool
        else:
            assert all(
                [cfg in kwargs for cfg in ["model_cfg", "layer_cfg"]]
            ), "All required ETNN `DictConfig`s must be provided."
            module_cfg = kwargs["model_cfg"]

            for k, v in module_cfg.items():
                setattr(self, k, v)

        self.core = ETNNCore(
            num_features_per_rank={0: self.in_dim0, 1: self.in_dim1, 2: self.in_dim2},
            num_hidden=self.emb_dim,
            num_out=self.emb_dim,
            num_layers=self.num_layers,
            adjacencies=["0_0_1", "0_0_2", "1_0", "2_0"],
            initial_features=[],
            visible_dims=[0, 1, 2],
            normalize_invariants=True,
            hausdorff_dists=True,
            batch_norm=True,
            sparse_invariant_computation=True,
            pos_update=False,
        )

        self.emb_0 = torch.nn.Linear(self.in_dim0, self.emb_dim)
        self.pool = get_aggregation(self.pool)

    def forward(self, batch):
        X = batch.pos

        device = X.device

        # ccc is taking way more time when compared to cell complex
        # ccc: CombinatorialComplex = batch.sse_cell_complex.to_combinatorial_complex()
        #
        # N2_0 = from_sparse(ccc.incidence_matrix(rank=0, to_rank=2).T)
        # N1_0 = from_sparse(ccc.incidence_matrix(rank=0, to_rank=1).T)
        # N0_0_via_1 = from_sparse(ccc.adjacency_matrix(rank=0, via_rank=1))
        # N0_0_via_2 = from_sparse(ccc.adjacency_matrix(rank=0, via_rank=2))
        if self.debug:
            cc: CellComplex = batch.sse_cell_complex
            Bt = [from_sparse(cc.incidence_matrix(rank=i, signed=False).T).to(device) for i in range(1,3)]
            N2_0 = (torch.sparse.mm(Bt[1], Bt[0]) / 2).coalesce()
            N1_0 = Bt[0].coalesce()
            N0_0_via_1 = from_sparse(cc.adjacency_matrix(rank=0, signed=False)).to(device)
            N0_0_via_2 = torch.sparse.mm(N2_0.T, N2_0).coalesce()
        else:
            N2_0 = batch.N2_0
            N1_0 = batch.N1_0
            N0_0_via_1 = batch.N0_0_via_1
            N0_0_via_2 = batch.N0_0_via_2

        def cell_list(i, format="list"):
            cell_index_2 = [torch.tensor(inner_list, device=device) for inner_list in batch.sse_cell_index]
            cell_index_1 = list(batch.edge_index.T)
            cell_index_0 = list(torch.arange(0, batch.x.shape[0], device=device).T.unsqueeze(1))
            list_of_cells = [
                cell_index_0,
                cell_index_1,
                cell_index_2
            ]
            return list_of_cells[i]

        batch.cell_list = cell_list

        def get_adj(neighborhood_matrix):
            return neighborhood_matrix.indices()[:, neighborhood_matrix.values() > 0]

        batch.adj_0_0_1 = get_adj(N0_0_via_1)
        batch.adj_0_0_2 = get_adj(N0_0_via_2)
        batch.adj_1_0 = get_adj(N1_0)
        batch.adj_2_0 = get_adj(N2_0)

        out = self.core(batch)
        return out

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_attr", "batch", "N2_0", "N1_0", "N0_0_via_1", "N0_0_via_2", "sse_attr"}


#%%
if __name__ == "__main__":
    #%%
    batch = torch.load("/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_featurised_batch_edge_processed_simple.pt", weights_only=False)
    print(batch)
    #%%
    model = ETNNModel(debug=True, activation="silu", num_layers=6, dropout=0, layer_cfg={
        "norm": "layer",
    })
    import time
    tik = time.time()
    out = model(batch)
    tok = time.time()
    print(f"Time taken: {tok-tik} seconds")
    #%%
    print(batch.pos)
    print(out['pos'])
    #%%

    assert torch.allclose(batch.pos, out['pos'], atol=10)
    assert torch.allclose(batch.pos, out['pos'], atol=1)
    print("All close")

