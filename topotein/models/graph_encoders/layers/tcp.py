from beartype import beartype as typechecker

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding, GCPLayerNorm, GCP
from typing import Any, Optional, Tuple, Union
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch
import torch
from jaxtyping import Bool, Float, Int64, jaxtyped


class TCPEmbedding(GCPEmbedding):
    def __init__(self, cell_input_dims, cell_hidden_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_input_dims = cell_input_dims
        self.cell_hidden_dims = cell_hidden_dims

        use_gcp_norm = kwargs.get("use_gcp_norm", True)
        nonlinearities = kwargs.get("nonlinearities", ("silu", "silu"))
        cfg = kwargs.get("cfg", None)

        self.cell_normalization = GCPLayerNorm(
            cell_input_dims if self.pre_norm else cell_hidden_dims,
            use_gcp_norm=use_gcp_norm
        )

        self.cell_embedding = GCP(
            cell_input_dims,
            cell_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
            self, batch: Union[Batch, ProteinBatch]
    ) -> Tuple[
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m chi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x xi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_cells c_hidden_dim"],
                Float[torch.Tensor, "batch_num_cells c rho_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_cells c_hidden_dim"],
        ],
    ]:
        node_rep, edge_rep = super().forward(batch)
        cell_rep = ScalarVector(batch.c, batch.rho)

        # TODO: calculate cell rep based on updated positions
        cell_rep = (
            cell_rep.scalar if not self.cell_embedding.vector_input_dim else cell_rep
        )

        if self.pre_norm:
            cell_rep = self.cell_normalization(cell_rep)

        cell_rep = self.cell_embedding(
            cell_rep,
            batch.sse_cell_index_simple,
            batch.f_ij_cell,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None),
        )

        if not self.pre_norm:
            cell_rep = self.cell_normalization(cell_rep)


        return node_rep, edge_rep, cell_rep


#%%

if __name__ == "__main__":
    from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding
    from proteinworkshop.models.utils import centralize, localize
    from omegaconf import OmegaConf
    from proteinworkshop.constants import PROJECT_PATH
    #%%
    print(PROJECT_PATH)
    #%%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch: ProteinBatch = torch.load(f'{PROJECT_PATH}/../test/data/sample_batch/sample_batch_for_tcp.pt', weights_only=False).to(device)


    batch.f_ij = localize(batch.pos, batch.edge_index)
    batch.f_ij_cell = localize(batch.pos, batch.sse_cell_index_simple)

    pos_centroid, X_c = centralize(batch, 'pos', batch.batch)
    ori_pos = batch.pos
    batch.pos = X_c
    batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = (
        batch.x,
        batch.x_vector_attr,
        batch.edge_attr,
        batch.edge_vector_attr,
        batch.sse_attr,
        batch.sse_vector_attr
    )

    cfg = OmegaConf.create({
        "r_max": 10.0,
        "num_rbf": 8,
        "scalar_gate": 0,
        "vector_gate": True,
        "enable_e3_equivariance": False,
        "nonlinearities": ["silu", "silu"],
        "pre_norm": False,
        "use_scalar_message_attention": True,
        "default_bottleneck": 4,
        "mp_cfg": {
            "edge_encoder": False,
            "edge_gate": False,
            "num_message_layers": 4,
            "message_residual": 0,
            "message_ff_multiplier": 1,
            "self_message": True
        },
        "scalar_nonlinearity": "silu",
        "vector_nonlinearity": "silu",
        "use_gcp_norm": True,
        "use_gcp_dropout": True,
        "num_feedforward_layers": 2
    })


    tcp_embedding = TCPEmbedding(
        node_input_dims=[49, 2],
        edge_input_dims=[9, 1],
        cell_input_dims=[15, 8],
        node_hidden_dims=[128, 16],
        edge_hidden_dims=[32, 4],
        cell_hidden_dims=[64, 8],
        cfg=cfg
    )
    (h, chi), (e, xi), (c, rho) = tcp_embedding(batch)
    print(f'h: {h.shape}, chi: {chi.shape}')
    print(f'e: {e.shape}, xi: {xi.shape}')
    print(f'c: {c.shape}, rho: {rho.shape}')


