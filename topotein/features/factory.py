from typing import List, Union

from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

from proteinworkshop.features.factory import ProteinFeaturiser, StructureRepresentation
from topotein.features.cell_features import compute_scalar_cell_features, compute_vector_cell_features
from topotein.features.cells import compute_cells
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
        cell_types: List[str],
        scalar_cell_features: List[ScalarCellFeature],
        vector_cell_features: List[VectorCellFeature],
    ):
        super(TopoteinFeaturiser, self).__init__(representation, scalar_node_features, vector_node_features, edge_types, scalar_edge_features, vector_edge_features)
        self.cell_types = cell_types
        self.scalar_cell_features = scalar_cell_features
        self.vector_cell_features = vector_cell_features

    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:

        for cell_type in self.cell_types:
            if cell_type.startswith("sse"):
                batch.sse = sse_onehot(batch)

        batch = super().forward(batch)

        # cells
        if self.cell_types:
            batch.cell_index, batch.cell_type, batch.cell_complex = compute_cells(
                batch, self.cell_types
            )
            batch.num_structure = len(self.cell_types)

        # Scalar cell features
        if self.scalar_cell_features:
            batch.cell_attr = compute_scalar_cell_features(
                batch, self.scalar_cell_features
            )

        # Vector cell features
        if self.vector_cell_features:
            batch = compute_vector_cell_features(
                batch, self.vector_cell_features
            )

        return batch

    def __repr__(self) -> str:
        return f"TopoteinFeaturiser(representation={self.representation}, scalar_node_features={self.scalar_node_features}, vector_node_features={self.vector_node_features}, edge_types={self.edge_types}, scalar_edge_features={self.scalar_edge_features}, vector_edge_features={self.vector_edge_features}, cell_types={self.cell_types}, scalar_cell_features={self.scalar_cell_features}, vector_cell_features={self.vector_cell_features})"
