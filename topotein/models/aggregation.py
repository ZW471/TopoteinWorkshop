from topomodelx import Aggregation
from topomodelx.utils.scatter import scatter


class Aggregator(Aggregation):
    def __init__(self, aggr_func="sum"):
        super().__init__(aggr_func=aggr_func, update_func=None)

    def update(self, inputs):
        # do not update in aggregator
        raise RuntimeWarning("Aggregator should not update")


class InterNeighborhoodAggregator(Aggregator):
    def __repr__(self):
        return f"InterNeighborhoodAggregator(aggr_func={self.aggr_func})"

class IntraNeighborhoodAggregator(Aggregator):
    def forward(self, neighborhood_matrix, x):
        return scatter(self.aggr_func)(x, neighborhood_matrix.indices()[0], dim=0, dim_size=neighborhood_matrix.size(0))
    def __repr__(self):
        return f"IntraNeighborhoodAggregator(aggr_func={self.aggr_func})"
