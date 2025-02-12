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
from topotein.models.graph_encoders.layers.ETNN import ETNNLayer


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


        self.layers = torch.nn.ModuleList([ETNNLayer(self.emb_dim, self.in_dim1, self.in_dim2, self.dropout, self.activation, **kwargs) for _ in range(self.num_layers)])
        self.emb_0 = torch.nn.Linear(self.in_dim0, self.emb_dim)
        self.pool = get_aggregation(self.pool)

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




        for layer in self.layers:
            H0, X = layer(X, H0, H1, H2, N0_0_via_1, N0_0_via_2, N2_0, N1_0)
        return {
            "node_embedding": H0,
            # "edge_embedding": torch.cat([H1, H1], dim=-1),
            # "sse_embedding": H2,
            "graph_embedding": self.pool(
                H0, batch.batch
            ),  # (n, d) -> (batch_size, d)
            "pos": X
        }

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

