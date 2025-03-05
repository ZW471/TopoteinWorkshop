import torch

from topotein.features.neighborhood_calculator import NeighborhoodCaculator
from topotein.features.topotein_neighborhood_calculator import TopoteinNeighborhoodCalculator
from topotein.features.topotein_complex import TopoteinComplex

def compute_neighborhoods(batch, neighborhoods):
    if isinstance(batch.sse_cell_complex, TopoteinComplex):
        nc = TopoteinNeighborhoodCalculator(batch.sse_cell_complex)
    else:
        nc = NeighborhoodCaculator(batch)
    return nc.calc_equations(neighborhoods)






if __name__ == "__main__":

    def are_sparse_tensors_equal(tensor1, tensor2):
        # Ensure both tensors are sparse
        if not tensor1.is_sparse or not tensor2.is_sparse:
            print("One or both tensors are not sparse.")
            return False

        # Check shapes
        if tensor1.size() != tensor2.size():
            print("Shapes do not match.")
            return False

        # Check indices
        if not torch.equal(tensor1.indices(), tensor2.indices()):
            print("Indices do not match.")
            return False

        # Check values
        if not torch.equal(tensor1.values(), tensor2.values()):
            print("Values do not match.")
            return False

        # If all checks pass, tensors are equal
        return True

    batch = torch.load("/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_featurised_batch_edge_processed_simple.pt", weights_only=False)
    neighborhoods = [
        'N2_0 = B2.T @ B1.T / 2',
        'N1_0 = B1.T',
        'N0_0_via_1 = A0',
        'N0_0_via_2 = B1 @ B2 @ B2.T @ B1.T / 4'
    ]

    neighborhoods = compute_neighborhoods(batch, neighborhoods)

    from toponetx import CellComplex
    from topomodelx.utils.sparse import from_sparse
    cc: CellComplex = batch.sse_cell_complex
    Bt = [from_sparse(cc.incidence_matrix(rank=i, signed=False).T) for i in range(1,3)]
    N2_0 = (torch.sparse.mm(Bt[1], Bt[0]) / 2).coalesce()
    N1_0 = Bt[0].coalesce()
    N0_0_via_1 = from_sparse(cc.adjacency_matrix(rank=0, signed=False))
    N0_0_via_2 = torch.sparse.mm(N2_0.T, N2_0).coalesce()

    print(N0_0_via_1)
    print(neighborhoods['N0_0_via_1'])

    assert are_sparse_tensors_equal(N2_0, neighborhoods['N2_0'])
    assert are_sparse_tensors_equal(N1_0, neighborhoods['N1_0'])
    assert are_sparse_tensors_equal(N0_0_via_1, neighborhoods['N0_0_via_1'])
    assert are_sparse_tensors_equal(N0_0_via_2, neighborhoods['N0_0_via_2'])




