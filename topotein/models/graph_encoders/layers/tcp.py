import torch_scatter
from beartype import beartype as typechecker
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding, GCPLayerNorm, GCP
from typing import Any, Optional, Tuple, Union
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch
import torch
from jaxtyping import Bool, Float, Int64, jaxtyped
from topotein.models.utils import centralize


class TCP(GCP):

    NODE_TYPE = "node"
    EDGE_TYPE = "edge"
    CELL_TYPE = "cell"

    def __init__(self, input_dims: ScalarVector, output_dims: ScalarVector, input_type: str,
                 nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
                 scalar_out_nonlinearity: Optional[str] = "silu", scalar_gate: int = 0, vector_gate: bool = True,
                 feedforward_out: bool = False, bottleneck: int = 1, scalarization_vectorization_output_dim: int = 3,
                 enable_e3_equivariance: bool = False, **kwargs):

        super().__init__(input_dims, output_dims, nonlinearities, scalar_out_nonlinearity, scalar_gate, vector_gate,
                         feedforward_out, bottleneck, scalarization_vectorization_output_dim, enable_e3_equivariance,
                         **kwargs)

        self.input_type = input_type

        if self.input_type == self.CELL_TYPE and self.vector_input_dim:
            scalar_vector_frame_dim = scalarization_vectorization_output_dim * 6
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim
                        + self.scalar_input_dim
                        + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                        ),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                    self.scalar_output_dim,
                    )
            )


    @jaxtyped(typechecker=typechecker)
    def scalarize(
        self,
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        enable_e3_equivariance: bool,
        dim_size: int,
        node_pos: Optional[Float[torch.Tensor, "n_nodes 3"]] = None,
        node_to_sse_mapping: torch.Tensor = None,
        node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None,
        cell_frames: Optional[Float[torch.Tensor, "batch_num_cells 3 3"]] = None,
    ) -> Float[torch.Tensor, "effective_batch_num_entities out_scalar_dim"]:
        row, col = edge_index[0], edge_index[1]
        input_type = self.input_type
        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        if input_type == self.NODE_TYPE:
            vector_rep_i = vector_rep[row]
        elif input_type == self.EDGE_TYPE:
            vector_rep_i = vector_rep
        elif input_type == self.CELL_TYPE:
            vector_rep_i = vector_rep  # leave for later
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        # project equivariant values onto corresponding local frames
        if vector_rep_i.ndim == 2:
            vector_rep_i = vector_rep_i.unsqueeze(-1)
        elif vector_rep_i.ndim == 3:
            vector_rep_i = vector_rep_i.transpose(-1, -2)

        if input_type == self.CELL_TYPE:
            edge_mask = None
            if node_mask is not None:
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                edge_index = edge_index[:, edge_mask]

            sse_mapping = node_to_sse_mapping
            sse_lookup = torch.ones(sse_mapping.size(0), dtype=torch.long, device=sse_mapping.device) * -1
            sse_lookup[sse_mapping.indices()[0]] = sse_mapping.indices()[1]
            cell_edge_index = torch.stack([sse_lookup[edge_index[i]] for i in range(2)], dim=0)

            mask = ((~(cell_edge_index == -1).any(dim=0)) & (cell_edge_index[0] != cell_edge_index[1]))  # mask of columns that don’t contain NaN, and no self-loops

            if edge_mask is not None:
                f_e_ij = frames[edge_mask][mask]
            else:
                f_e_ij = frames[mask]

            cell_edge_index = cell_edge_index[:, mask]  # c -> c
            edge_index = edge_index[:, mask]  # n -> n

            # print(cell_edge_index.shape, edge_index.shape)

            f_c_i = cell_frames[cell_edge_index[0]]
            f_c_j = cell_frames[cell_edge_index[1]]

            vector_rep_i = vector_rep_i[cell_edge_index[0]]
            # print(vector_rep_i)

            fff = torch.einsum('bij,bjk,bkl->bil', f_c_i, f_e_ij, f_c_j)  # = torch.bmm(torch.bmm(f_c_i, f_e_ij), f_c_j)
            local_scalar_rep_i = torch.bmm(vector_rep_i, fff)

            node_pos_in_sse = node_pos[edge_index[0]]
            com, _ = centralize(
                node_pos_in_sse, 
                cell_edge_index[0], 
                node_mask=node_mask[edge_index[0]] if node_mask is not None else None
            )
            com_lifted = com[cell_edge_index[0]]
            r_com_i = node_pos_in_sse - com_lifted
            local_torque_rep_i = torch.cross(r_com_i.unsqueeze(-1), torch.bmm(vector_rep_i, fff), dim=1)

            local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2).reshape(vector_rep_i.shape[0], 9)
            local_torque_rep_i = local_torque_rep_i.transpose(-1, -2).reshape(vector_rep_i.shape[0], 9)

            if enable_e3_equivariance:
                raise NotImplementedError('E3 equivariance is not yet implemented for cell inputs.')

        elif input_type == self.NODE_TYPE or input_type == self.EDGE_TYPE:
            if node_mask is not None:
                edge_mask = node_mask[row] & node_mask[col]
                local_scalar_rep_i = torch.zeros(
                    (edge_index.shape[1], 3, 3), device=edge_index.device
                )
                local_scalar_rep_i[edge_mask] = torch.matmul(
                    frames[edge_mask], vector_rep_i[edge_mask]
                )
                local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
            else:
                local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

            # potentially enable E(3)-equivariance and, thereby, chirality-invariance
            if enable_e3_equivariance:
                # avoid corrupting gradients with an in-place operation
                local_scalar_rep_i_copy = local_scalar_rep_i.clone()
                local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
                local_scalar_rep_i = local_scalar_rep_i_copy

            # reshape frame-derived geometric scalars
            local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if input_type == self.NODE_TYPE:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            return torch_scatter.scatter(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )
        elif input_type == self.EDGE_TYPE:
            return local_scalar_rep_i
        elif input_type == self.CELL_TYPE:
            return torch.cat([torch_scatter.scatter(
                rep_i,
                cell_edge_index[0],
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            ) for rep_i in [local_scalar_rep_i, local_torque_rep_i]], dim=1)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

    # noinspection PyMethodOverriding
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"],
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        cell_frames: Optional[Float[torch.Tensor, "batch_num_cells 3 3"]] = None,
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        node_pos: Optional[Float[torch.Tensor, "n_nodes 3"]] = None,
        node_to_sse_mapping: Optional[Int64[torch.Tensor, "n_nodes batch_num_cells"]] = None,
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"],
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask,
                cell_frames=cell_frames,
                node_pos=node_pos,
                node_to_sse_mapping=node_to_sse_mapping,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)






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

        self.cell_embedding = TCP(
            cell_input_dims,
            cell_hidden_dims,
            input_type=TCP.CELL_TYPE,
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

        node_to_sse_mapping = getattr(batch, "node_to_sse_mapping", None)

        cell_rep = self.cell_embedding(
            cell_rep,
            batch.edge_index,
            frames=batch.f_ij,
            cell_frames=batch.f_ij_cell,
            node_mask=getattr(batch, "mask", None),
            node_pos=batch.pos,
            node_to_sse_mapping=node_to_sse_mapping
        )

        if not self.pre_norm:
            cell_rep = self.cell_normalization(cell_rep)


        return node_rep, edge_rep, cell_rep


#%%

if __name__ == "__main__":
    from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding
    from proteinworkshop.models.utils import localize, safe_norm, get_activations
    from omegaconf import OmegaConf, DictConfig
    from proteinworkshop.constants import PROJECT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch: ProteinBatch = torch.load(f'{PROJECT_PATH}/../test/data/sample_batch/sample_batch_for_tcp.pt', weights_only=False).to(device)
    print(batch)


    batch.f_ij = localize(batch.pos, batch.edge_index)
    batch.f_ij_cell = localize(batch.pos, batch.sse_cell_index_simple)
    batch.node_to_sse_mapping = batch.N2_0.T.coalesce()

    pos_centroid, X_c = centralize(batch.pos, batch.batch)
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


