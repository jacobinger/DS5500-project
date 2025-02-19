import gzip
from rdkit import Chem
import torch
from torch_geometric.data import Data

def mol_to_graph(mol):
    """
    Convert an RDKit molecule to a PyTorch Geometric Data object.
    """
    if mol is None:
        return None
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum(), atom.GetDegree()])
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    
    if mol.HasProp("ACTIVITY"):
        try:
            activity = float(mol.GetProp("ACTIVITY"))
        except ValueError:
            activity = 0.0  # handle conversion errors
        data.y = torch.tensor([activity], dtype=torch.float)
    else:
        # Optionally, skip molecules without the desired property
        # or assign a default value:
        data.y = torch.tensor([0.0], dtype=torch.float)

    
    return data


def load_dataset(sdf_path, max_mols=None):
    """
    Load molecules from a (possibly compressed) SDF file and convert them to graph Data objects.
    :param sdf_path: Path to the SDF file (e.g., 'chembl_35.sdf.gz').
    :param max_mols: Optional limit to the number of molecules (for testing).
    :return: List of PyTorch Geometric Data objects.
    """
    dataset = []
    
    # Handle gzipped files
    if sdf_path.endswith('.gz'):
        with gzip.open(sdf_path, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            for i, mol in enumerate(suppl):
                if mol is None:
                    continue
                data = mol_to_graph(mol)
                if data is None:
                    continue
                # (Optional) Set a dummy label if you don't have one.
                # data.y = torch.tensor([0.0], dtype=torch.float)
                dataset.append(data)
                if max_mols and (i + 1) >= max_mols:
                    break
    else:
        suppl = Chem.SDMolSupplier(sdf_path)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            data = mol_to_graph(mol)
            if data is None:
                continue
            dataset.append(data)
            if max_mols and (i + 1) >= max_mols:
                break
    return dataset

# Example usage:
sdf_file = 'data/chembl_35.sdf.gz'  # Update this to your file location.
dataset = load_dataset(sdf_file, max_mols=1000)  # Using a subset for quick testing.
print(f"Loaded {len(dataset)} molecules from the SDF file.")


from torch_geometric.loader import DataLoader

batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MoleculeGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(MoleculeGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # For demonstration, we use one output unit (e.g., regression).
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # First convolution + activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second convolution
        x = self.conv2(x, edge_index)
        # Aggregate node features to get a molecule-level representation
        x = global_mean_pool(x, batch)
        # Final prediction layer
        x = self.fc(x)
        return x

import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MoleculeGNN(num_node_features=2, hidden_channels=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Use CrossEntropyLoss for classification tasks

def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Check if data.y exists; if not, create a dummy target.
        if data.y is None:
            target = torch.zeros((data.num_graphs, 1), device=device, dtype=torch.float)
        else:
            # Ensure target has the shape [batch_size, 1]
            target = data.y.view(-1, 1).to(device)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


# Training loop
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    loss = train()
    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
