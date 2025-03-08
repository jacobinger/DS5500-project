import os
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

# Paths to SMILES files
smi_files = [
    os.path.join("zinc_data", "BA/AARN/BAAARN.smi"),
    os.path.join("zinc_data", "BA/ABRN/BAABRN.smi"),
    os.path.join("zinc_data", "BB/AARN/BBAARN.smi"),
    os.path.join("zinc_data", "BB/ABRN/BBABRN.smi"),
    os.path.join("zinc_data", "BC/AARN/BCAARN.smi"),
]

screening_ligands = {}
compound_count = 0
invalid_smiles_count = 0  # Track invalid SMILES for debugging

# Create a Morgan fingerprint generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

for smi_file in smi_files:
    if os.path.exists(smi_file):
        with open(smi_file, "r") as f:
            lines = f.read().splitlines()
            # Skip the first line if it looks like a header
            start_idx = 1 if lines and lines[0].startswith("smiles") else 0
            zinc_smiles = [line.split()[0] for line in lines[start_idx:] if line.strip()]
        
        for smiles in zinc_smiles:
            if compound_count >= 10000:
                break
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = morgan_gen.GetFingerprint(mol)
                screening_ligands[f"ZINC_{compound_count}"] = np.array(list(fp), dtype=np.float32)
                compound_count += 1
            else:
                invalid_smiles_count += 1
    else:
        print(f"File not found: {smi_file}")

print(f"Generated {len(screening_ligands)} screening ligands")
print(f"Skipped {invalid_smiles_count} invalid SMILES strings")

# Save the fingerprints for later use
import pickle
with open("screening_ligands.pkl", "wb") as f:
    pickle.dump(screening_ligands, f)
    
import torch
from torch_geometric.data import HeteroData
import pickle

# Load stored ligand fingerprints
with open("screening_ligands.pkl", "rb") as f:
    screening_ligands = pickle.load(f)

# Convert to tensor (ensure correct dtype)
ligand_features = torch.tensor(np.array(list(screening_ligands.values()), dtype=np.float32))

# ✅ Create PyG HeteroData
data = HeteroData()

# ✅ Store ligand node features
data['ligand'].x = ligand_features  # Ensures ligand exists

# Debugging: Check data structure
print("✅ Ligand feature tensor shape:", data['ligand'].x.shape)  # Should be (10000, 1024)
