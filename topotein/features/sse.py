from beartype import beartype as typechecker
import pydssp
from graphein.protein.tensor import Protein
from graphein.protein.tensor.data import ProteinBatch
import torch


@typechecker
def annotate_protein_sses_pydssp(protein: Protein) -> torch.Tensor:
    """Annotate secondary structure elements for a protein using PyTorch. Much faster than P-SEA!"""

    return pydssp.assign(protein.coords[:, :4, :], out_type='onehot').float()

@typechecker
def sse_onehot(protein_batch: ProteinBatch) -> torch.Tensor:
    """Annotate secondary structure elements for a protein using PyTorch. Much faster than P-SEA!"""

    return torch.cat(protein_batch.protein_apply(annotate_protein_sses_pydssp))

@typechecker
def get_sse_cell_group(sse_batch: torch.Tensor, protein_batch: torch.Tensor, num_of_classes: int = 3, minimal_group_size: int = 3, simple_version: bool = False):
    """
    sse_batch: A (b, num_of_classes) tensor of one-hot rows.
    protein_batch: A (b,) tensor indicating the protein each SSE belongs to.
    num_of_classes: Number of classes in the one-hot dimension (default=3).
    minimal_group_size: Minimum size of the group to be included.

    Returns:
        A dictionary mapping each class 'c' -> list of consecutive index groups
        where sse_batch[i] == one-hot vector for class 'c' and belongs to the same protein.

        For example, if num_of_classes=3,
        you get {
          0: [[start_index, ..., end_index], [start_index, ..., end_index], ...],
          1: [...],
          2: [...]
        }
    """
    # --- Helper function to find consecutive runs of True in a 1D boolean mask ---
    def find_consecutive_true_indices(mask: torch.Tensor, least_consecutive_length: int = 3, simple_version: bool = False):
        """
        mask: (b,) boolean tensor.
        returns: A list of lists, each sub-list is a consecutive run of indices where mask == True.
        """
        if not mask.any():
            return []

        # Convert boolean mask to int (0 or 1)
        m = mask.int()
        # torch.diff is available in PyTorch >= 1.7; emulate if needed:
        dm = m[1:] - m[:-1]  # shape (b-1,)

        # Where dm == 1 => start of a run; where dm == -1 => end of a run
        starts = (dm == 1).nonzero(as_tuple=True)[0] + 1
        ends = (dm == -1).nonzero(as_tuple=True)[0]

        # Edge case: if first element is True, prepend index 0 to 'starts'
        if m[0].item() == 1:
            starts = torch.cat([torch.tensor([0], device=starts.device), starts])
        # Edge case: if last element is True, append last index to 'ends'
        if m[-1].item() == 1:
            ends = torch.cat([ends, torch.tensor([m.shape[0] - 1], device=ends.device)])

        # Pair up starts and ends
        starts_list = starts.tolist()
        ends_list = ends.tolist()

        groups = []
        for s, e in zip(starts_list, ends_list):
            if e - s < least_consecutive_length:  # Ensure group meets the minimum size
                continue
            if simple_version:
                groups.append((s, e))
            else:
                groups.append(tuple(range(s, e + 1)))
        return groups

    # --- Main logic: for each class, find where sse_batch == one-hot vector of that class ---
    device = sse_batch.device
    all_groups = {}
    # We'll create a one-hot vector for each class c, then compare
    for c in range(num_of_classes):
        # one-hot vector for class c
        one_hot_c = torch.zeros(num_of_classes, device=device, dtype=sse_batch.dtype)
        one_hot_c[c] = 1

        # Build mask: True where row == that one-hot
        mask_c = (sse_batch == one_hot_c).all(dim=1)  # shape (b,)

        # Ensure SSEs belong to the same protein
        protein_ids = protein_batch.unique()
        group_list = []
        for protein in protein_ids:
            protein_mask = protein_batch == protein  # Mask for SSEs of this protein
            combined_mask = mask_c & protein_mask  # Combine with the class mask

            # Find all consecutive runs of True in the combined mask
            groups_c = find_consecutive_true_indices(combined_mask, least_consecutive_length=minimal_group_size, simple_version=simple_version)
            group_list.extend(groups_c)

        all_groups[c] = group_list

    return all_groups


