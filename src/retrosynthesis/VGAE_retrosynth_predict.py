import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)
from utils.molecule_utils import graph_to_molecule

class MoleculeDBDataset(InMemoryDataset):
    def __init__(self, db_path, limit=1000, transform=None, pre_transform=None):
        self.db_path = db_path
        self.limit = limit
        super(MoleculeDBDataset, self).__init__('.', transform, pre_transform)
        torch.serialization.add_safe_globals([
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage
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
        # This should have been created during training.
        raise NotImplementedError("Dataset processing not implemented in generate.py. Run train.py first.")

# Define the same GCNEncoder used during training.
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

def generate_n_molecules(n=10, num_nodes=6):
    """
    Generates n novel molecules using the pre-trained VGAE model, with an adjustable number of nodes per molecule.
    
    Parameters:
      n (int): Number of molecules to generate.
      num_nodes (int): Number of nodes (atoms) to include in each generated molecule.
    
    Returns:
      List of RDKit Mol objects. If a molecule cannot be generated,
      the corresponding list element will be None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset to compute latent space statistics.
    db_path = os.path.join("data", "chembl_35.db")
    dataset = MoleculeDBDataset(db_path, limit=1000)
    
    # Define model parameters (should match training)
    in_channels = 2
    hidden_channels = 64
    latent_dim = 32
    
    encoder = GCNEncoder(in_channels, hidden_channels, latent_dim)
    model = VGAE(encoder).to(device)
    
    # Load the trained VGAE model weights.
    model_dir = "models"
    model_path = os.path.join(model_dir, "vgae_molecule.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Compute latent space statistics from the dataset.
    z_list = []
    x_list = []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            mu, logvar = model.encoder(data.x, data.edge_index)
            z = model.reparametrize(mu, logvar)
            z_list.append(z)
            x_list.append(data.x)
    z_all = torch.cat(z_list, dim=0)
    x_all = torch.cat(x_list, dim=0)
    
    z_mean = torch.mean(z_all, dim=0)
    z_std = torch.std(z_all, dim=0)
    
    generated_molecules = []
    for _ in range(n):
        # Sample a new latent vector with adjustable number of nodes.
        z = z_mean + torch.randn(num_nodes, latent_dim).to(device) * z_std * 0.1
        
        # Create a complete graph edge_index for the sampled nodes.
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
        adj_recon = model.decoder(z, edge_index, sigmoid=True)
        edge_mask = adj_recon > 0.7
        sampled_edge_index = edge_index[:, edge_mask]
        
        # Infer node features from the training distribution.
        unique, counts = torch.unique(x_all[:, 0], return_counts=True)
        probs = counts.float() / counts.sum()
        sampled_atomic_nums = torch.multinomial(probs, num_nodes, replacement=True)
        x = torch.zeros(num_nodes, 2)
        x[:, 0] = unique[sampled_atomic_nums]
        x[:, 1] = 2  # Placeholder for additional features.
        
        sampled_graph = Data(x=x, edge_index=sampled_edge_index)
        mol = graph_to_molecule(sampled_graph)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            print("Generated molecule SMILES:", smiles)
        else:
            print("Failed to create a valid molecule.")
        generated_molecules.append(mol)
    
    return generated_molecules


#test function
# if __name__ == "__main__":
#     molecules = generate_n_molecules(n=5)
#     # Optionally, display/save images for each generated molecule.
#     for i, mol in enumerate(molecules):
#         if mol:
#             img = Draw.MolToImage(mol)
#             image_path = f"generated_molecule_{i}.png"
#             img.save(image_path)
#             img.show()
