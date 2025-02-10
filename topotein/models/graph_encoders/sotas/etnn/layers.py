from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from topotein.models.graph_encoders.sotas.etnn import utils, invariants


class ETNNLayer(nn.Module):
    def __init__(
        self,
        adjacencies: List[str],
        visible_dims: list[int],
        num_hidden: int,
        num_features_map: dict[str, int],
        batch_norm: bool = False,
        lean: bool = True,
        pos_update: bool = False,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.num_features_map = num_features_map
        self.visible_dims = visible_dims
        self.batch_norm = batch_norm
        self.lean = lean
        self.pos_update = pos_update

        # messages
        self.message_passing = nn.ModuleDict(
            {
                adj: BaseMessagePassingLayer(
                    num_hidden,
                    self.num_features_map[adj],
                    batch_norm=batch_norm,
                    lean=lean,
                )
                for adj in adjacencies
            }
        )

        # state update
        self.update = nn.ModuleDict()
        for dim in self.visible_dims:
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            update_layers = [nn.Linear(factor * num_hidden, num_hidden)]
            if self.batch_norm:
                update_layers.append(nn.BatchNorm1d(num_hidden))
            if not self.lean:
                extra_layers = [nn.SiLU(), nn.Linear(num_hidden, num_hidden)]
                if self.batch_norm:
                    extra_layers.append(nn.BatchNorm1d(num_hidden))
                update_layers.extend(extra_layers)
            self.update[str(dim)] = nn.Sequential(*update_layers)

        # position update
        if pos_update:
            self.pos_update_wts = nn.Linear(num_hidden, 1, bias=False)
            nn.init.trunc_normal_(self.pos_update_wts.weight, std=0.02)

    def radial_pos_update(
        self, pos: Tensor, mes: dict[str, Tensor], adj: dict[str, Tensor]
    ) -> Tensor:
        # find the key corresponding to the 0_0_x adjacency
        key = [k for k in adj if k[0] == "0" and k[2] == "0"][0]
        send, recv = adj[key]
        wts = self.pos_update_wts(mes[key][recv])

        # collect the pos_delta for each node: going from
        # [num_edges, num_hidden] to [num_nodes, num_hidden]
        delta = utils.scatter_add(
            (pos[send] - pos[recv]) * wts, send, dim=0, dim_size=pos.size(0)
        )
        return pos + 0.1 * delta

    def forward(
        self,
        x: Dict[str, Tensor],
        adj: Dict[str, Tensor],
        inv: Dict[str, Tensor],
        pos: Tensor,
    ) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=adj[adj_type],
                edge_attr=inv[adj_type],
            )
            for adj_type in self.adjacencies
        }

        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature]
                + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim],
                dim=1,
            )
            for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        if self.pos_update:
            pos = self.radial_pos_update(pos, mes, adj)

        return x, pos


class BaseMessagePassingLayer(nn.Module):
    def __init__(
        self, num_hidden, num_inv, batch_norm: bool = False, lean: bool = True
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.lean = lean
        message_mlp_layers = [
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
        ]
        if self.batch_norm:
            message_mlp_layers.insert(1, nn.BatchNorm1d(num_hidden))

        if not self.lean:
            extra_layers = [
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
            ]
            if self.batch_norm:
                extra_layers.insert(1, nn.BatchNorm1d(num_hidden))
            message_mlp_layers.extend(extra_layers)
        self.message_mlp = nn.Sequential(*message_mlp_layers)
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = utils.scatter_add(
            messages * edge_weights, index_rec, dim=0, dim_size=x_rec.shape[0]
        )

        return messages_aggr


#%%

if __name__ == "__main__":
    #%%
    batch = torch.load(
        "/data/sample_batch/sample_featurised_batch_edge_processed_simple.pt",
        weights_only=False,
    )
    print(batch)
    #%%
    from toponetx import CellComplex
    from topomodelx.utils.sparse import from_sparse

    batch.to('cpu')
    X = batch.pos         # positions
    H0 = batch.x          # 0-dimensional features
    H1 = batch.edge_attr  # 1-dimensional features (dense)
    H2 = batch.sse_attr   # 2-dimensional features (dense)
    device = X.device
    emb = torch.randn(57, 512).to(device)
    H0 = H0 @ emb

    cc: CellComplex = batch.sse_cell_complex
    Bt = [from_sparse(cc.incidence_matrix(rank=i, signed=False).T).to(device)
          for i in range(1, 3)]
    N2_0 = (torch.sparse.mm(Bt[1], Bt[0]) / 2).coalesce()
    N1_0 = Bt[0].coalesce()
    N0_0_via_1 = from_sparse(cc.adjacency_matrix(rank=0, signed=False)).to(device)
    N0_0_via_2 = torch.sparse.mm(N2_0.T, N2_0).coalesce()

    #%%
    # Create the layer (note: the expected adjacency types are strings like "A_B_C"
    # where A is the source and C the target dimension)
    layer = ETNNLayer(
        adjacencies=["0_0_1", "0_0_2", "1_0", "2_0"],
        visible_dims=[0, 1, 2],
        num_hidden=512,
        num_features_map={"0_0_1": 512, "0_0_2": 512, "1_0": 512, "2_0": 512},
        batch_norm=True,
        lean=True,
        pos_update=True,
    )

    import time

    def get_index(adj):
        # If adj is a sparse tensor, return its indices (dense tensor of shape [2, nnz])
        if isinstance(adj, torch.Tensor) and adj.is_sparse:
            return adj.indices()
        return adj

    # Pre-process the adjacency matrices for the layer.
    # For each adjacency type, we convert any sparse tensor into its indices.
    adj = {
        "0_0_1": get_index(N0_0_via_1),
        "0_0_2": get_index(N0_0_via_2),
        "1_0": get_index(N1_0),
        "2_0": get_index(N2_0),
    }

    # For the invariants (edge features), use the dense features from the batch.
    # (In the original code these were H1 and H2.)
    inv_fun = invariants.compute_invariants


    inv = {
        "0_0_1": X,
        "0_0_2": H2,
        "1_0": H1,  # connecting dimension 1 to 0 uses H1
        "2_0": H2,  # connecting dimension 2 to 0 uses H2
    }

    tik = time.time()
    H, pos = layer(
        x={"0": H0, "1": H1, "2": H2},
        adj=adj,
        inv=inv,
        pos=X,
    )
    tok = time.time()
    print(f"Time taken: {tok-tik:.2f}s")

    # Now test equivariance/invariance.
    Q = torch.randn(3, 3)
    t = torch.rand(3)
    posQt = pos @ Q + t

    # If the layer is equivariant, applying Q and t to the positions and node features
    # should yield transformed outputs. We can re-use the same adjacencies and invariants.
    # (Note: In this example, the adjacencies are independent of Q and t.)
    adj_transformed = {
        "0_0_1": get_index(N0_0_via_1),
        "0_0_2": get_index(N0_0_via_2),
        "1_0": get_index(N1_0),
        "2_0": get_index(N2_0),
    }

    QtH, QtPos = layer(
        x={"0": H0, "1": H1, "2": H2},
        adj=adj_transformed,
        inv=inv,
        pos=X @ Q + t,
    )

    # The hidden state is expected to be invariant (within a tolerance) and the positions equivariant.
    assert torch.allclose(H, QtH, atol=10), f"Hidden state is not invariant to Q and t\n{H}\n{QtH}"
    assert torch.allclose(posQt, QtPos, atol=10), f"Position is not equivariant to Q and t\n{posQt}\n{QtPos}"
    assert torch.allclose(posQt, QtPos, atol=1), f"Position is not equivariant to Q and t\n{posQt}\n{QtPos}"
    assert torch.allclose(posQt, QtPos, atol=.1), f"Position is not equivariant to Q and t\n{posQt}\n{QtPos}"

    print("All tests passed")

    #%%
    print(pos)
    print(X)

    #%%
    print((pos - X).abs().max())
    #%%
    print((pos - X).abs().mean())
    #%%
    print((pos - X).mean())