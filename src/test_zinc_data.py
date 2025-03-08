import torch
from torch_geometric.data.storage import BaseStorage

# ✅ Allow safe unpickling for PyG data
torch.serialization.add_safe_globals([BaseStorage])

# ✅ Load the full dataset
data = torch.load("hetero_data.pt", weights_only=False)

# ✅ Move data to device before accessing properties
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ✅ Debugging: Ensure ligand exists
print("🔍 Node types in data:", data.node_types)
if 'ligand' in data.node_types:
    print("✅ Ligand feature shape:", data['ligand'].x.shape)
else:
    raise ValueError("❌ ERROR: 'ligand' node type is missing from data!")
