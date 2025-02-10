import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from topotein.models.graph_encoders.sotas.etnn.layers import ETNNLayer
from topotein.models.graph_encoders.sotas.etnn import utils, invariants


class ETNNCore(nn.Module):
    """
    The E(n)-Equivariant Topological Neural Network (ETNN) model.
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        hausdorff_dists: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.0,
        lean: bool = True,
        global_pool: bool = False,  # whether or not to use global pooling
        sparse_invariant_computation: bool = False,
        sparse_agg_max_cells: int = 100,  # maximum size to consider for diameter and hausdorff dists
        pos_update: bool = False,  # performs the equivariant position update, optional
    ) -> None:
        super().__init__()

        self.initial_features = initial_features

        # make inv_fts_map for backward compatibility
        self.num_invariants = 5 if hausdorff_dists else 3
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        max_dim = max(num_features_per_rank.keys())
        self.global_pool = global_pool
        self.visible_dims = visible_dims
        self.pos_update = pos_update
        self.dropout = dropout

        # params for invariant computation
        self.sparse_invariant_computation = sparse_invariant_computation
        self.sparse_agg_max_cells = sparse_agg_max_cells
        self.hausdorff = hausdorff_dists
        self.cell_list_fmt = "list" if sparse_invariant_computation else "padded"

        if sparse_invariant_computation:
            self.inv_fun = invariants.compute_invariants_sparse
        else:
            self.inv_fun = invariants.compute_invariants

        # keep only adjacencies that are compatible with visible_dims
        if visible_dims is not None:
            self.adjacencies = []
            for adj in adjacencies:
                max_rank = max(int(rank) for rank in adj.split("_")[:2])
                if max_rank in visible_dims:
                    self.adjacencies.append(adj)
        else:
            self.visible_dims = list(range(max_dim + 1))
            self.adjacencies = adjacencies

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj], affine=False)
                    for adj in self.adjacencies
                }
            )

        embedders = {}
        for dim in self.visible_dims:
            embedder_layers = [nn.Linear(num_features_per_rank[dim], num_hidden)]
            if self.batch_norm:
                embedder_layers.append(nn.BatchNorm1d(num_hidden))
            embedders[str(dim)] = nn.Sequential(*embedder_layers)
        self.feature_embedding = nn.ModuleDict(embedders)

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    self.batch_norm,
                    self.lean,
                    self.pos_update,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in visible_dims:
            if self.global_pool:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_hidden),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_hidden)
            else:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_out),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_out)

        if self.global_pool:
            self.post_pool = nn.Sequential(
                nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_out),
            )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device

        cell_ind = {
            str(i): graph.cell_list(i, format=self.cell_list_fmt)
            for i in self.visible_dims
        }

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        # compute initial features
        # features = {}
        # for feature_type in self.initial_features:
        #     features[feature_type] = {}
        #     for i in self.visible_dims:
        #         if feature_type == "node":
        #             features[feature_type][str(i)] = invariants.compute_centroids(
        #                 cell_ind[str(i)], graph.x
        #             )
        #         elif feature_type == "mem":
        #             mem = {i: getattr(graph, f"mem_{i}") for i in self.visible_dims}
        #             features[feature_type][str(i)] = mem[i].float()
        #         elif feature_type == "hetero":
        #             features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        # x = {
        #     str(i): torch.cat(
        #         [
        #             features[feature_type][str(i)]
        #             for feature_type in self.initial_features
        #         ],
        #         dim=1,
        #     )
        #     for i in self.visible_dims
        # }

        x = {
            '0': graph.x,
            '1': graph.edge_attr,
            '2': graph.sse_attr,
        }

        # if using sparse invariant computation, obtain indces
        inv_comp_kwargs = {
            "cell_ind": cell_ind,
            "adj": adj,
            "hausdorff": self.hausdorff,
        }
        if self.sparse_invariant_computation:
            agg_indices, _ = invariants.sparse_computation_indices_from_cc(
                cell_ind, adj, self.sparse_agg_max_cells
            )
            inv_comp_kwargs["rank_agg_indices"] = agg_indices

        # embed features and E(n) invariant information
        pos = graph.pos
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}
        inv = self.inv_fun(pos, **inv_comp_kwargs)

        if self.normalize_invariants:
            inv = {
                adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()
            }

        # message passing
        for layer in self.layers:
            x, pos = layer(x, adj, inv, pos)
            if self.pos_update:
                inv = self.inv_fun(pos, **inv_comp_kwargs)
                if self.normalize_invariants:
                    inv = {
                        adj: self.inv_normalizer[adj](feature)
                        for adj, feature in inv.items()
                    }
            # apply dropout if needed
            if self.dropout > 0:
                x = {
                    dim: nn.functional.dropout(feature, p=self.dropout)
                    for dim, feature in x.items()
                }

        # read out
        out = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        if self.global_pool:
            # create one dummy node with all features equal to zero for each graph and each rank
            cell_batch = {
                str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"])
                for i in self.visible_dims
            }
            out = {
                dim: global_add_pool(out[dim], cell_batch[dim])
                for dim, feature in out.items()
            }
            state = torch.cat(
                tuple([feature for dim, feature in out.items()]),
                dim=1,
            )
            out = self.post_pool(state)
            out = torch.squeeze(out, -1)

        return {
            "node_embedding": out['0'],
            "edge_embedding": out['1'],
            "sse_embedding": out['2'],
            "graph_embedding": None,
            "pos": pos
        }

    def __str__(self):
        return f"ETNN ({self.type})"


#%%

if __name__ == "__main__":
    #%%
    batch = torch.load(
        "/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_featurised_batch_directed2.pt",
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
    emb = torch.randn(49, 512).to(device)
    H0 = H0 @ emb

    cc: CellComplex = batch.sse_cell_complex
    Bt = [from_sparse(cc.incidence_matrix(rank=i, signed=False).T).to(device)
          for i in range(1, 3)]
    N2_0 = (torch.sparse.mm(Bt[1], Bt[0]) / 2).coalesce()
    N1_0 = Bt[0].coalesce()
    N0_0_via_1 = from_sparse(cc.adjacency_matrix(rank=0, signed=False)).to(device)
    N0_0_via_2 = torch.sparse.mm(N2_0.T, N2_0).coalesce()

    #%%



    cell_index_2 = [torch.tensor(inner_list) for inner_list in batch.sse_cell_index]
    cell_index_1 = list(batch.edge_index.T)
    cell_index_0 = list(torch.arange(0, batch.x.shape[0]).T.unsqueeze(1))
    def cell_list(i, format="padded"):
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

    #%%



    model = ETNNCore(
        num_features_per_rank={0: 49, 1: 2, 2: 4},
        num_hidden=512,
        num_out=512,
        num_layers=6,
        adjacencies=["0_0_1", "0_0_2", "1_0", "2_0"],
        initial_features=["node"],
        visible_dims=[0, 1, 2],
        normalize_invariants=True,
        hausdorff_dists=False,
        sparse_invariant_computation=True,
        pos_update=False,
    )
    #%%
    import time
    tik = time.time()
    out = model(
        batch
    )
    print(f"Time taken: {time.time() - tik}")
    print(out)

    #%%
    (out['pos'] - batch.pos).abs().max()

    from utils.visualization import show_protein, output_to_color, color_batch
    from utils.batch import batch_reset_sse_cell_complex_color

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(batch)
    batch_reset_sse_cell_complex_color(batch)
    color_batch(batch, output_to_color({
        "node": batch.x,
        "edge": torch.cat([batch.edge_attr, batch.edge_attr], dim=-1),
        "sse": batch.sse_attr
    }))
    show_protein(batch, 15, color_edge=False)

    #%%
    from utils.visualization import plot_color_3d_distribution
    def eval_model(model, batch):
        original_pos = batch.pos.clone()
        out = model(batch)
        del out["graph_embedding"]
        for key, value in out.items():
            print(key, value.shape)
        colored_out = output_to_color(out)
        batch.pos = out["pos"]
        color_batch(batch, colored_out)
        show_protein(batch, 15, color_edge=False, show_sse=True)
        # plot_color_3d_distribution(colored_out)
        batch.pos = original_pos

    eval_model(model, batch)