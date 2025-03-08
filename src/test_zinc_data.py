import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn.functional as F

# Load the data with weights_only=False
data = torch.load("hetero_data.pt", weights_only=False)

print("🔍 Node types in data.x_dict:", data.x_dict.keys())
if 'ligand' in data.x_dict:
    print("✅ Ligand feature shape:", data.x_dict['ligand'].shape)
else:
    raise ValueError("❌ ERROR: 'ligand' node type is missing from data.x_dict!")

# ✅ Step 1: Ensure ligand self-loops exist
num_ligands = data.x_dict['ligand'].shape[0]
self_edges = torch.arange(num_ligands).repeat(2, 1).to(torch.long)

# Explicitly add ligand self-loops
data.edge_index_dict[('ligand', 'self', 'ligand')] = self_edges
print("✅ Forced self-loops for 'ligand', edge count:", self_edges.shape)

# ✅ Step 2: Print all edge types before message passing
print("\n📌 Full edge structure BEFORE message passing:")
for edge_type, edge_tensor in data.edge_index_dict.items():
    print(f"  {edge_type}: Shape {edge_tensor.shape}")

# ✅ Step 3: Define the model
class TestHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('ligand', 'interacts', 'protein'): SAGEConv((-1, -1), hidden_channels),
            ('protein', 'interacts', 'protein'): SAGEConv((-1, -1), hidden_channels),
            ('ligand', 'self', 'ligand'): SAGEConv((-1, -1), hidden_channels)  # ✅ Ensures ligand gets updated
        })
        self.proj = Linear(input_dim, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        print("\n🔥 Running HeteroConv")
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # ✅ Debug: Check if ligand exists
        print("🚀 Updated x_dict keys:", x_dict.keys())  

        if 'ligand' in x_dict:
            ligand_out = x_dict['ligand'].mean(dim=0)
        else:
            print("❌ 'ligand' is STILL missing! Using original features instead.")
            ligand_out = self.proj(data.x_dict['ligand'].mean(dim=0))  

        return self.lin(ligand_out)

# ✅ Step 4: Run the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

ligand_feature_dim = data.x_dict['ligand'].shape[1]
model = TestHeteroGNN(hidden_channels=64, input_dim=ligand_feature_dim).to(device)

print("\n📌 Existing edge types before forward pass:", data.edge_index_dict.keys())

output = model(data)
print(f"\n✅ Model output: {output}")
