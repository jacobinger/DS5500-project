import torch
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn.functional as F

# ✅ Allow safe unpickling for PyG data BEFORE loading
torch.serialization.add_safe_globals([BaseStorage])

# ✅ Load the full dataset
data = torch.load("hetero_data.pt", weights_only=False)

# ✅ Move to device before accessing properties
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ✅ Debugging: Ensure ligand & protein exist
print("🔍 Node types in data:", data.node_types)
if 'ligand' in data.node_types and 'protein' in data.node_types:
    print("✅ Ligand feature shape:", data['ligand'].x.shape)
    print("✅ Protein feature shape:", data['protein'].x.shape)
else:
    raise ValueError("❌ ERROR: Missing required node types in data!")

# ✅ Step 2: Print edge types before message passing
print("\n📌 Edge types in data:", data.edge_types)
for edge_type in data.edge_types:
    print(f"  {edge_type}: Shape {data[edge_type].edge_index.shape}")

# ✅ Step 3: Define the model
class TestHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('ligand', 'interacts', 'protein'): SAGEConv((-1, -1), hidden_channels),
            ('protein', 'interacts', 'ligand'): SAGEConv((-1, -1), hidden_channels)  # ✅ Ensures ligand gets messages
        })
        self.proj = Linear(input_dim, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        print("\n🔥 Running HeteroConv")

        # ✅ Print edges before passing to HeteroConv
        print("\n📌 Edge types being passed to HeteroConv:")
        for edge_type in data.edge_types:
            print(f"  {edge_type}: Shape {data[edge_type].edge_index.shape}")

        x_dict = self.conv1(data.x_dict, {etype: data[etype].edge_index for etype in data.edge_types})
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # ✅ Debug: Check if ligand exists after message passing
        print("🚀 Updated x_dict keys:", x_dict.keys())

        if 'ligand' in x_dict:
            print("✅ 'ligand' successfully updated!")
            ligand_out = x_dict['ligand'].mean(dim=0)
        else:
            print("❌ 'ligand' is STILL missing! Using original features instead.")
            ligand_out = self.proj(data['ligand'].x.mean(dim=0))  

        return self.lin(ligand_out)

# ✅ Step 4: Run the model
ligand_feature_dim = data['ligand'].x.shape[1]
model = TestHeteroGNN(hidden_channels=64, input_dim=ligand_feature_dim).to(device)

print("\n📌 Running model forward pass...")
output = model(data)
print(f"\n✅ Model output: {output}")
