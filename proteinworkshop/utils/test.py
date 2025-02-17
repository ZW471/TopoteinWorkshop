import random

from graphein.protein.tensor import Protein

PDB_LIST = ["3eiy", "1hcn", "9dlx", "8zdd"]
PDB_LIST_SAFE = []


def get_random_protein_safe(pdb=None) -> "Protein":
    """Utility/testing function to get a random proteins."""
    if pdb is None:
        pdbs = PDB_LIST_SAFE
        pdb = random.choice(pdbs)
    return Protein().from_pdb_code(pdb)


for pdb in PDB_LIST:
    try:
        get_random_protein_safe(pdb=pdb)
        PDB_LIST_SAFE.append(pdb)
    except ValueError:
        print(f"Error with pdb code {pdb} in random protein function, removed.")