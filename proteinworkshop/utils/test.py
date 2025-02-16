import random

from graphein.protein.tensor import Protein


def get_random_protein_safe() -> "Protein":
    """Utility/testing function to get a random proteins."""
    pdbs = ["3eiy", "4hhb", "1hcn"]
    pdb = random.choice(pdbs)
    return Protein().from_pdb_code(pdb)