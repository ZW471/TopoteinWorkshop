from typing import List, Union

from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

from proteinworkshop.features.factory import ProteinFeaturiser, StructureRepresentation
from topotein.features.cell_features import compute_scalar_cell_features, compute_vector_cell_features
from topotein.features.cells import compute_sses
from topotein.features.sse import sse_onehot
from proteinworkshop.types import ScalarNodeFeature, VectorNodeFeature, ScalarEdgeFeature, VectorEdgeFeature, \
    ScalarCellFeature, VectorCellFeature


class TopoteinFeaturiser(ProteinFeaturiser):
    def __init__(
        self,
        representation: StructureRepresentation,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
        sse_types: List[str],
        scalar_sse_features: List[ScalarCellFeature],
        vector_sse_features: List[VectorCellFeature],
    ):
        super(TopoteinFeaturiser, self).__init__(representation, scalar_node_features, vector_node_features, edge_types, scalar_edge_features, vector_edge_features)
        self.sse_types = sse_types
        self.scalar_sse_features = scalar_sse_features
        self.vector_sse_features = vector_sse_features

    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:

        batch.sse = sse_onehot(batch)  # this is node sse

        batch = super().forward(batch)

        # cells
        if self.sse_types:
            # sse: onehot type of sse for groups
            # sse_cell_index: cells that represents sses
            # sse_cell_complex: the whole cell complex that contains structural information of this higher-order graph
            batch.sse, batch.sse_cell_index, batch.sse_cell_complex = compute_sses(
                batch, self.sse_types
            )
            batch.num_sse_type = len(self.sse_types)

        # Scalar cell features
        if self.scalar_sse_features:
            batch.sse_attr = compute_scalar_cell_features(
                batch, self.scalar_sse_features
            )

        # Vector cell features
        if self.vector_sse_features:
            batch.sse_vector_attr = compute_vector_cell_features(
                batch, self.vector_sse_features
            )

        return batch

    def __repr__(self) -> str:
        return f"TopoteinFeaturiser(representation={self.representation}, scalar_node_features={self.scalar_node_features}, vector_node_features={self.vector_node_features}, edge_types={self.edge_types}, scalar_edge_features={self.scalar_edge_features}, vector_edge_features={self.vector_edge_features}, cell_types={self.sse_types}, scalar_cell_features={self.scalar_sse_features}, vector_cell_features={self.vector_sse_features})"
