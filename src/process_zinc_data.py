import os
import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import HeteroData

# ✅ Paths
ZINC20_DIR = "zinc_data"
OUTPUT_PATH = "data/zinc20_hetero_graph.pt"
CHEMBL_GRAPH_PATH = "data/chembl_35_hetero_graph_PD.pt"

# ✅ Ensure ChEMBL 35 graph exists (for target features)
if not os.path.exists(CHEMBL_GRAPH_PATH):
    raise FileNotFoundError(f"ChEMBL graph not found at {CHEMBL_GRAPH_PATH}")

chembl_graph = torch.load(CHEMBL_GRAPH_PATH, weights_only=False)
target_features = chembl_graph['target'].x  # Use ChEMBL targets

# ✅ Load ZINC20 SMILES Files
smi_files = [
    os.path.join(ZINC20_DIR, "BA/AARN/BAAARN.smi"),
    os.path.join(ZINC20_DIR, "BA/ABRN/BAABRN.smi"),
    os.path.join(ZINC20_DIR, "BB/AARN/BBAARN.smi"),
    os.path.join(ZINC20_DIR, "BB/ABRN/BBABRN.smi"),
    os.path.join(ZINC20_DIR, "BC/AARN/BCAARN.smi"),
]

screening_ligands = {}
compound_count = 0
invalid_smiles_count = 0

# ✅ Morgan Fingerprint Generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

# ✅ Process Ligand Data
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
        print(f"❌ File not found: {smi_file}")

print(f"✅ Generated {len(screening_ligands)} screening ligands")
print(f"✅ Skipped {invalid_smiles_count} invalid SMILES strings")

# ✅ Save Ligand Fingerprints for Later Use
with open("data/screening_ligands.pkl", "wb") as f:
    pickle.dump(screening_ligands, f)

# ✅ Load Ligand Data from Pickle
with open("data/screening_ligands.pkl", "rb") as f:
    screening_ligands = pickle.load(f)

# ✅ Convert to Tensor (Ensure at least 1 ligand exists)
if len(screening_ligands) > 0:
    ligand_features = torch.tensor(np.array(list(screening_ligands.values()), dtype=np.float32))
else:
    print("❌ No valid ligands were found in ZINC20 data.")
    ligand_features = torch.empty((0, 1024))  # Empty tensor to prevent errors

# ✅ Create HeteroData Object
data = HeteroData()

# ✅ Step 1: Store Ligand Features
data['ligand'].x = ligand_features
data['ligand'].num_nodes = ligand_features.shape[0]

print("✅ Ligand feature tensor shape:", data['ligand'].x.shape)
print("🔍 Node types in data:", list(data.node_types))

# ✅ Step 2: Store Target Features (from ChEMBL 35)
data['target'].x = target_features
data['target'].num_nodes = target_features.shape[0]

print("✅ Added target node features:", data['target'].x.shape)

# ✅ Step 3: Generate Ligand-Target Interactions (Only if Ligands Exist)
num_ligands = data['ligand'].num_nodes
num_targets = data['target'].num_nodes

if num_ligands > 0:
    num_edges = min(10000, num_ligands * num_targets)  # Limit interactions
    ligand_indices = torch.randint(0, num_ligands, (num_edges,))
    target_indices = torch.randint(0, num_targets, (num_edges,))
    
    ligand_target_edges = torch.stack([ligand_indices, target_indices], dim=0)
    data[('ligand', 'binds_to', 'target')].edge_index = ligand_target_edges

    # ✅ Reverse Edge
    target_ligand_edges = torch.stack([target_indices, ligand_indices], dim=0)
    data[('target', 'binds_to', 'ligand')].edge_index = target_ligand_edges

    print("✅ Generated bidirectional ligand-target edges.")
else:
    print("❌ No ligands found, skipping edge generation.")

# ✅ Step 4: Save Processed Data
torch.save(data, OUTPUT_PATH)
print(f"\n✅ Successfully saved ZINC20 hetero graph at {OUTPUT_PATH}")


##CUT##

