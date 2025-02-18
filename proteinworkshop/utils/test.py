import random
import torch
from graphein.protein.tensor import Protein
import proteinworkshop.constants
import os
import warnings

PDB_LIST = ["3eiy", "1hcn", "8py0", "8vth"]


def get_random_protein_safe(pdb=None, offline=True) -> "Protein":
    """Utility/testing function to get a random proteins."""
    if pdb is None:
        pdb = random.choice(PDB_LIST)

    if offline:
        try:
            return torch.load(os.path.join(proteinworkshop.constants.DATA_PATH, "sample_proteins", f"{pdb}.pt"), weights_only=False)
        except FileNotFoundError:
            return get_random_protein_safe(pdb=pdb, offline=False)
    else:

        print("downloading pdb files from internet - this can cause issues sometimes!")
        protein = Protein().from_pdb_code(pdb)

        data_dir = proteinworkshop.constants.DATA_PATH
        sample_proteins_dir = os.path.join(data_dir, "sample_proteins")
        
        # Ensure the sample_proteins directory exists
        os.makedirs(sample_proteins_dir, exist_ok=True)
        
        print(f'Saving protein {pdb} to {data_dir}...')
        
        
        torch.save(protein, os.path.join(data_dir, "sample_proteins", f"{pdb}.pt"))

        return protein



if __name__ == "__main__":

    data_dir = proteinworkshop.constants.DATA_PATH
    print(data_dir)

    for pdb in PDB_LIST:
        print(pdb)
        try:
            protein = get_random_protein_safe(pdb=pdb, offline=False)
            print(protein)
        except Exception:
            print(f"Error with pdb code {pdb} in random protein function, removed.")
            continue

        print('saving...')
        torch.save(protein, os.path.join(data_dir, "sample_proteins", f"{pdb}.pt"))