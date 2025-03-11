import os
import sqlite3
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage  # Import the missing type
from torch_geometric.nn import GCNConv, VGAE

# Adjust sys.path to find your utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)

from utils.molecule_utils import smiles_to_graph

# =============================================================================
# Dataset: Build a dataset by querying the ChEMBL database for SMILES
# =============================================================================

class MoleculeDBDataset(InMemoryDataset):
    def __init__(self, db_path, limit=1000, transform=None, pre_transform=None):
        self.db_path = db_path
        self.limit = limit
        super(MoleculeDBDataset, self).__init__('.', transform, pre_transform)
        # Add safe globals before loading the processed file.
        torch.serialization.add_safe_globals([
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage  # Add this to allowlist torch_geometric's GlobalStorage
        ])
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        # Connect to the ChEMBL database.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query: Extract canonical SMILES from compound_structures.
        query = """
        SELECT canonical_smiles
        FROM compound_structures
        WHERE canonical_smiles IS NOT NULL
        LIMIT ?
        """
        cursor.execute(query, (self.limit,))
        rows = cursor.fetchall()
        
        for (smiles,) in rows:
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            data_list.append(graph)
        
        conn.close()
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


# =============================================================================
# VGAE Model Components: Define a GCN-based encoder.
# =============================================================================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# =============================================================================
# Training Loop for VGAE
# =============================================================================
def train_vgae(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        # The reconstruction loss tries to recover the graph connectivity.
        loss = model.recon_loss(z, data.edge_index)
        # Add KL divergence loss (scaled by inverse of number of nodes).
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =============================================================================
# Main Function: Prepare dataset, train model, and sample from the latent space.
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your ChEMBL database file.
    db_path = os.path.join("data/chembl_35.db")  # Adjust if needed.
    
    # Build the dataset from the database.
    dataset = MoleculeDBDataset(db_path, limit=1000)  # Increase limit as needed.
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Assume each molecule node has two features: for example, atomic number and degree.
    # (Ensure that your `smiles_to_graph` function creates node features accordingly.)
    in_channels = 2
    hidden_channels = 64
    latent_dim = 32
    
    encoder = GCNEncoder(in_channels, hidden_channels, latent_dim)
    model = VGAE(encoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train_vgae(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Save the trained VGAE model.
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "vgae_molecule.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Example: Sample from the latent space and decode to obtain a reconstructed graph.
    # Example: Sample from the latent space and decode to obtain a reconstructed graph.
    model.eval()
    with torch.no_grad():
        # Define the number of nodes for the sampled graph
        num_nodes = 10  # Adjust based on your desired graph size (e.g., average nodes in your dataset)
        
        # Sample a latent vector for `num_nodes` nodes
        z = torch.randn(num_nodes, latent_dim).to(device)  # Shape: [num_nodes, latent_dim]
        
        # Generate an edge_index for all possible edges (fully connected graph)
        # This creates pairs for every possible edge (undirected graph)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
        # Shape: [2, num_nodes * (num_nodes - 1) / 2]
        
        # Decode to get the reconstructed adjacency probabilities
        adj_recon = model.decoder(z, edge_index, sigmoid=True)
        print("Sampled latent vector, reconstructed adjacency probabilities:")
        print(adj_recon.cpu())  # Shape: [num_edges], probabilities between 0 and 1
        
        # Optionally, threshold to create a binary adjacency matrix
        threshold = 0.5
        adj_binary = (adj_recon > threshold).float()
        print("Binary adjacency matrix (thresholded):")
        print(adj_binary.cpu())

if __name__ == "__main__":
    main()
