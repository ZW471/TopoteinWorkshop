from typing import List, Union

from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

from proteinworkshop.features.factory import ProteinFeaturiser, StructureRepresentation
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
        scaler_cell_features: List[ScalarCellFeature],
        vector_cell_features: List[VectorCellFeature],
    ):
        super(TopoteinFeaturiser, self).__init__(representation, scalar_node_features, vector_node_features, edge_types, scalar_edge_features, vector_edge_features)
        self.cell_typeset = cell_types
        self.scaler_cell_features = scaler_cell_features
        self.vector_cell_features = vector_cell_features

    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:

        for cell_type in self.cell_typeset:
            if cell_type.startswith("sse"):
                batch.sse = sse_onehot(batch)

        batch = super().forward(batch)

        # TODO: add logic for attaching cells, maybe see how the edges are constructed

        return batch

    def __repr__(self) -> str:
        return f"TopoteinFeaturiser(representation={self.representation}, scalar_node_features={self.scalar_node_features}, vector_node_features={self.vector_node_features}, edge_types={self.edge_types}, scalar_edge_features={self.scalar_edge_features}, vector_edge_features={self.vector_edge_features}, cell_types={self.cell_typeset}, scaler_cell_features={self.scaler_cell_features}, vector_cell_features={self.vector_cell_features})"
