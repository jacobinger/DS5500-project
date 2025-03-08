import os
import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import HeteroData

# ✅ Step 1: Process ZINC Data and Generate Ligand Features
smi_files = [
    os.path.join("zinc_data", "BA/AARN/BAAARN.smi"),
    os.path.join("zinc_data", "BA/ABRN/BAABRN.smi"),
    os.path.join("zinc_data", "BB/AARN/BBAARN.smi"),
    os.path.join("zinc_data", "BB/ABRN/BBABRN.smi"),
    os.path.join("zinc_data", "BC/AARN/BCAARN.smi"),
]

screening_ligands = {}
compound_count = 0
invalid_smiles_count = 0

# Create a Morgan fingerprint generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

for smi_file in smi_files:
    if os.path.exists(smi_file):
        with open(smi_file, "r") as f:
            lines = f.read().splitlines()
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

print(f"✅ Generated {len(screening_ligands)} screening ligands")
print(f"✅ Skipped {invalid_smiles_count} invalid SMILES strings")

# ✅ Save ligand fingerprints for later use
with open("screening_ligands.pkl", "wb") as f:
    pickle.dump(screening_ligands, f)

# ✅ Step 2: Convert Ligand Data to PyG Format
with open("screening_ligands.pkl", "rb") as f:
    screening_ligands = pickle.load(f)

# Convert to tensor
ligand_features = torch.tensor(np.array(list(screening_ligands.values()), dtype=np.float32))

# ✅ Step 3: Create a Proper `HeteroData` Object
data = HeteroData()

# ✅ Step 4: Store Ligand Features in `HeteroData`
data['ligand'].x = ligand_features
data['ligand'].num_nodes = ligand_features.shape[0]

print("✅ Ligand feature tensor shape:", data['ligand'].x.shape)
print("🔍 Node types in data:", list(data.node_types))

# ✅ Step 5: Ensure Protein Has Features
num_proteins = 500  # Choose a reasonable number based on your dataset
protein_features = torch.randn((num_proteins, 1024))  # Randomly initialize protein features
data['protein'].x = protein_features
data['protein'].num_nodes = num_proteins

print("✅ Added protein node features:", data['protein'].x.shape)

# ✅ Step 6: Ensure Bidirectional Edges (ligand ↔ protein)
num_ligands = data['ligand'].num_nodes  # Total ligand count
num_proteins = data['protein'].num_nodes  # Total protein count

# ✅ Generate valid edges within bounds
ligand_indices = torch.randint(0, num_ligands, (100,))  # Ensure within ligand range
protein_indices = torch.randint(0, num_proteins, (100,))  # Ensure within protein range

ligand_protein_edges = torch.stack([ligand_indices, protein_indices], dim=0)
data[('ligand', 'interacts', 'protein')].edge_index = ligand_protein_edges

# ✅ Reverse edge: Ensure valid indices
protein_ligand_edges = torch.stack([protein_indices, ligand_indices], dim=0)
data[('protein', 'interacts', 'ligand')].edge_index = protein_ligand_edges

print("✅ Ensured all edges reference valid node indices.")

# ✅ Step 7: Save Processed Data
torch.save(data, "hetero_data.pt")
print("\n✅ Successfully saved `hetero_data.pt`")
