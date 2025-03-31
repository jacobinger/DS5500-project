import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, VGAE
from torch_geometric.data import HeteroData
from rdkit.Chem import rdMolDescriptors
import pickle


# Set up project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
from create_new_drugs.VGAE_affinity_utils import generate_n_molecules
from utils.molecule_utils import smiles_to_graph

# Output directories
output_dir = "png/novel"
os.makedirs(output_dir, exist_ok=True)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Models ===
class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=32, target_in_channels=1280, hidden_channels=256, dropout_p=0.2):
        super().__init__()
        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((ligand_in_channels, target_in_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((target_in_channels, ligand_in_channels), hidden_channels)
        }, aggr='sum')
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        }, aggr='sum')
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x_dict = {'ligand': data['ligand'].x, 'target': data['target'].x}
        edge_index_dict = {
            ('ligand', 'binds_to', 'target'): data['ligand', 'binds_to', 'target'].edge_index,
            ('target', 'binds_to', 'ligand'): data['ligand', 'binds_to', 'target'].edge_index.flip(0)
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.bn1(torch.relu(self.dropout(x))) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.bn2(torch.relu(self.dropout(x))) for key, x in x_dict.items()}
        edge_index = data['ligand', 'binds_to', 'target'].edge_index
        ligand_feats = x_dict['ligand'][edge_index[0]]
        target_feats = x_dict['target'][edge_index[1]]
        edge_feats = torch.cat([ligand_feats, target_feats], dim=-1)
        out = self.edge_predictor(edge_feats).squeeze(-1)
        return out

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# === Utility functions ===
def molecule_to_ligand_features(mol, radius=2, n_bits=1280):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr).unsqueeze(0)


def prepare_evaluation_data(mol, target_features):
    ligand_features = molecule_to_ligand_features(mol)
    data = HeteroData()
    data['ligand'].x = ligand_features
    data['target'].x = target_features
    data['ligand', 'binds_to', 'target'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return data

def evaluate_generated_molecule(mol, gnn_model, device, target_features, threshold=0.5, use_sigmoid=True):
    gnn_model.eval()
    data = prepare_evaluation_data(mol, target_features)
    data = data.to(device)
    with torch.no_grad():
        output = gnn_model(data)
        score_tensor = output if isinstance(output, torch.Tensor) else output['score']
        score = torch.sigmoid(score_tensor).item() if use_sigmoid else score_tensor.item()
    return score, score > threshold


# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    node_counts = [5, 6, 7, 8, 9]
    molecules_per_count = 3
    all_generated_molecules = {}
    results = []

    for num_nodes in node_counts:
        print(f"\nGenerating molecules with {num_nodes} nodes:")
        generated_molecules = generate_n_molecules(n=molecules_per_count, num_nodes=num_nodes)
        all_generated_molecules[num_nodes] = generated_molecules

    # Load VGAE model
    vgae_encoder = GCNEncoder(in_channels=2, hidden_channels=64, latent_dim=32)
    vgae_model = VGAE(vgae_encoder).to(device)
    vgae_model.load_state_dict(torch.load("models/vgae_molecule.pth", map_location=device))
    vgae_model.eval()

    # Load HeteroGNN model
    gnn_model = HeteroGNN(
        ligand_in_channels=1280,
        target_in_channels=1280,
        hidden_channels=256,
        dropout_p=0.2
    ).to(device)

    gnn_model.load_state_dict(torch.load("data/sageconv_model.pt", map_location=device))

    gnn_model.eval()

    # Generate random target feature vector (replace with actual if available)
    with open(os.path.join("data", "target_dict.pkl"), "rb") as f:
        target_dict = pickle.load(f)

    target_id = list(target_dict.keys())[0]
    target_tensor = target_dict[target_id][1]  # numpy array
    target_features = torch.tensor(target_tensor).unsqueeze(0).to(device)

    for num_nodes, molecules in all_generated_molecules.items():
        print(f"\nEvaluating molecules with {num_nodes} nodes:")
        for i, mol in enumerate(molecules):
            print(f"\nMolecule {i+1} (nodes: {num_nodes}):")
            if mol:
                smiles = Chem.MolToSmiles(mol)
                print("SMILES:", smiles)
                score, viable = evaluate_generated_molecule(mol, gnn_model, device, target_features)
                print(f"Predicted binding affinity score: {score:.4f}")
                print(f"=> Molecule is predicted to be a {'good binder' if viable else 'poor binder'}.")

                img = Draw.MolToImage(mol)
                image_path = os.path.join(output_dir, f"generated_molecule_{num_nodes}_nodes_{i}.png")
                img.save(image_path)
                print(f"Saved image to {image_path}")

                results.append({
                    "SMILES": smiles,
                    "Score": score
                })
            else:
                print("Failed to generate a valid molecule.")
                results.append({
                    "SMILES": None,
                    "Score": None
                })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join("data", "generated_molecule_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

if __name__ == "__main__":
    main()
