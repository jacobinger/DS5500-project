import os
import sqlite3
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GCNConv
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)

from utils.molecule_utils import smiles_to_graph

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # Aggregate node features to obtain a graph-level prediction.
        return x.mean(dim=0)

# Create a dataset that loads data from the ChEMBL database.
class ChEMBLDataset(InMemoryDataset):
    def __init__(self, db_path, transform=None, pre_transform=None):
        self.db_path = db_path
        super(ChEMBLDataset, self).__init__('.', transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # Not used because we process from the database.
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # No download needed – data is local.
        pass
    
    def process(self):
        data_list = []
        # Connect to the ChEMBL database.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query to fetch canonical SMILES and an activity value (e.g., IC50).
        query = """
        SELECT cs.canonical_smiles, a.standard_value
        FROM compound_structures cs
        JOIN activities a ON cs.molregno = a.molregno
        WHERE a.standard_value IS NOT NULL
          AND a.standard_type = 'IC50'
        LIMIT 1000
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for smiles, value in rows:
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            try:
                # Convert the activity to a float.
                target = float(value)
            except ValueError:
                continue
            # Attach the target as a graph label.
            graph.y = torch.tensor([target], dtype=torch.float)
            data_list.append(graph)
        
        conn.close()
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

# Training loop
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Assuming a regression task with MSE loss.
        loss = loss_fn(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    # Set device (GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your ChEMBL database file.
    db_path = os.path.join("chembl_35.db")
    
    # Create the dataset from the database.
    dataset = ChEMBLDataset(db_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the model.
    # In our case, each node has two features (atomic number and degree).
    model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train(model, loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Save the trained model for later use.
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gcn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
