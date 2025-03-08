import subprocess
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Example: Download a single tranche file using curl
tranche_url = "http://files.docking.org/3D/AA/AARA/AAAARA.smi.gz"  # Adjust based on your selection
output_file = "AAAARA.smi.gz"

try:
    subprocess.run(["curl", "-o", output_file, tranche_url], check=True)

    # Read SMILES from gzip file
    with gzip.open(output_file, "rt") as f:
        zinc_smiles = [line.split()[0] for line in f.read().splitlines()[:10000]]  # Limit to 10k

    # Generate fingerprints
    screening_ligands = {}
    for i, smiles in enumerate(zinc_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            screening_ligands[f"ZINC_{i}"] = np.array(fp, dtype=np.float32)

    print(f"Generated {len(screening_ligands)} screening ligands")

except subprocess.CalledProcessError as e:
    print(f"Failed to download file: {e}")
