import os
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator  # Import the new generator
import numpy as np

# Paths to SMILES files
smi_files = [
    os.path.join("zinc_data", "BA/AARN/BAAARN.smi"),
    os.path.join("zinc_data", "BA/ABRN/BAABRN.smi"),
    os.path.join("zinc_data", "BB/AARN/BBAARN.smi"),
    # Add more files to reach 10,000 compounds
    os.path.join("zinc_data", "BB/ABRN/BBABRN.smi"),
    os.path.join("zinc_data", "BC/AARN/BCAARN.smi"),
]

screening_ligands = {}
compound_count = 0

# Create a Morgan fingerprint generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

for smi_file in smi_files:
    if os.path.exists(smi_file):
        with open(smi_file, "r") as f:
            zinc_smiles = [line.split()[0] for line in f.read().splitlines()]
        
        for smiles in zinc_smiles:
            if compound_count >= 10000:
                break
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Use the new MorganGenerator
                fp = morgan_gen.GetFingerprint(mol)
                screening_ligands[f"ZINC_{compound_count}"] = np.array(list(fp), dtype=np.float32)
                compound_count += 1
    else:
        print(f"File not found: {smi_file}")

print(f"Generated {len(screening_ligands)} screening ligands")

# Save the fingerprints for later use
import pickle
with open("screening_ligands.pkl", "wb") as f:
    pickle.dump(screening_ligands, f)