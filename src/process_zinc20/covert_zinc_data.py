import pickle
import torch
import numpy as np
from torch_geometric.data import HeteroData

# Load the fingerprints
with open("screening_ligands.pkl", "rb") as f:
    screening_ligands = pickle.load(f)

# Initialize the heterogeneous graph
data = HeteroData()

# Add ligand nodes (one node per ligand)
num_ligands = len(screening_ligands)
ligand_features = np.array([screening_ligands[f"ZINC_{i}"] for i in range(num_ligands)])  # Convert to NumPy array first
data['ligand'].x = torch.tensor(ligand_features, dtype=torch.float32)  # Shape: (10000, 1024)

# Add protein nodes (placeholder - replace with your protein data)
num_protein_nodes = 100
protein_features = torch.randn(num_protein_nodes, 128)  # Placeholder
data['protein'].x = protein_features

# Add ligand-protein interaction edges (placeholder - replace with real interactions)
edge_index_ligand_to_protein = torch.tensor(
    [[i, i % num_protein_nodes] for i in range(min(num_ligands, num_protein_nodes))],
    dtype=torch.long
).t()  # Shape: (2, 100)
data['ligand', 'interacts', 'protein'].edge_index = edge_index_ligand_to_protein

# Add protein-protein edges (e.g., residue contacts - placeholder)
edge_index_protein_to_protein = torch.tensor(
    [[i, i+1] for i in range(num_protein_nodes-1)],
    dtype=torch.long
).t()  # Shape: (2, 99)
data['protein', 'interacts', 'protein'].edge_index = edge_index_protein_to_protein

# Optional: Add edge attributes (e.g., binding scores) if you have them
edge_weights = torch.ones(edge_index_ligand_to_protein.size(1), dtype=torch.float32)
data['ligand', 'interacts', 'protein'].edge_weight = edge_weights

print(data)

# Save the data for training
torch.save(data, "hetero_data.pt")