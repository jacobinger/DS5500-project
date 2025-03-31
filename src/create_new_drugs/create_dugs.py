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
from rdkit.Chem import QED, Descriptors
from tqdm import tqdm
#import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs

# Set up project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
from create_new_drugs.VGAE_affinity_utils import generate_n_molecules
from src.utils.molecule_utils import smiles_to_graph
from src.utils.molecule_utils import graph_to_molecule
from src.train_VGAE_model.train_VGAE_model import MoleculeDBDataset


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
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
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

def get_top_scoring_molecules(generated_molecules, gnn_model, target_features, device, top_k=5):
    results = []
    for mol in generated_molecules:
        if mol is None:
            continue
        try:
            score, _ = evaluate_generated_molecule(mol, gnn_model, device, target_features)
            results.append((mol, score))
        except Exception as e:
            print(f"Scoring failed: {e}")
    
    results.sort(key=lambda tup: tup[1], reverse=True)  # Sort by score descending
    return results[:top_k]

def optimize_latent_vector(
    z_init,
    gnn_model,
    target_features,
    device,
    model_decoder,
    unique_atoms,
    atom_probs,
    steps=20,
    lr=0.05,
    qed_thresh=0.5,
    mw_range=(150, 500)
):
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    best_score = -float("inf")
    best_mol = None
    score_trajectory = []

    for step in range(steps):
        optimizer.zero_grad()

        num_nodes = z.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(device)
        adj_recon = model_decoder(z, edge_index, sigmoid=True)
        edge_mask = adj_recon > 0.6
        sampled_edge_index = edge_index[:, edge_mask]

        sampled_atomic_nums = torch.multinomial(atom_probs, num_nodes, replacement=True)
        x = torch.zeros(num_nodes, 2, device=device)
        x[:, 0] = unique_atoms[sampled_atomic_nums]
        x[:, 1] = 2

        graph = Data(x=x, edge_index=sampled_edge_index)
        mol = graph_to_molecule(graph)

        if mol is None:
            continue

        try:
            qed = QED.qed(mol)
            mw = Descriptors.MolWt(mol)
            if qed < qed_thresh or not (mw_range[0] <= mw <= mw_range[1]):
                continue

            score, _ = evaluate_generated_molecule(mol, gnn_model, device, target_features)
            score_trajectory.append(score)
            loss = -torch.tensor(score, requires_grad=True)
            loss.backward()
            optimizer.step()

            if score > best_score:
                best_score = score
                best_mol = mol

        except Exception as e:
            print(f"Optimization error at step {step}: {e}")
            continue

    return best_mol, best_score, score_trajectory


# === Main ===
def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    node_counts = [5, 6, 7, 8, 9]
    molecules_per_count = 200
    top_k = 5
    results = []
    optimized_results = []

    # Load models
    gnn_model = HeteroGNN(1280, 1280, 256, 0.2).to(device)
    gnn_model.load_state_dict(torch.load("data/sageconv_model.pt", map_location=device))
    gnn_model.eval()

    vgae_encoder = GCNEncoder(2, 64, 32)
    vgae_model = VGAE(vgae_encoder).to(device)
    vgae_model.load_state_dict(torch.load("models/vgae_molecule.pth", map_location=device))
    vgae_model.eval()

    # Prepare latent statistics
    dataset = MoleculeDBDataset("data/train_2_graphs.db", limit=1000)
    z_list, x_list = [], []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            mu, logvar = vgae_model.encoder(data.x, data.edge_index)
            z = vgae_model.reparametrize(mu, logvar)
            z_list.append(z)
            x_list.append(data.x)
    z_all = torch.cat(z_list, dim=0)
    x_all = torch.cat(x_list, dim=0)
    z_mean = torch.mean(z_all, dim=0)
    z_std = torch.std(z_all, dim=0)
    unique, counts = torch.unique(x_all[:, 0], return_counts=True)
    probs = counts.float() / counts.sum()

    target_features = torch.rand(1, 1280, device=device)

    # Greedy filtering phase
    for num_nodes in node_counts:
        print(f"\nGenerating {molecules_per_count} molecules with {num_nodes} nodes:")
        generated = generate_n_molecules(molecules_per_count, num_nodes)
        top_candidates = get_top_scoring_molecules(generated, gnn_model, target_features, device, top_k)
        for i, (mol, score) in enumerate(top_candidates):
            if mol:
                smiles = Chem.MolToSmiles(mol)
                path = f"png/novel/top_candidate_{num_nodes}_nodes_{i}.png"
                Draw.MolToImage(mol).save(path)
                results.append({"SMILES": smiles, "Score": score, "Node_Count": num_nodes, "Image_Path": path})
                print(f"Top {i+1}: {smiles} | Score: {score:.4f}")

    # Optimization phase
    print("\nStarting latent space optimization...")
    for num_nodes in node_counts:
        for i in tqdm(range(10), desc=f"Optimizing with {num_nodes} nodes"):
            z_init = z_mean + torch.randn(num_nodes, z_mean.shape[0]).to(device) * z_std * 0.5
            try:
                mol, score, trajectory = optimize_latent_vector(
                    z_init, gnn_model, target_features, device, vgae_model.decoder,
                    unique, probs, steps=25, lr=0.1, qed_thresh=0.5, mw_range=(150, 500)
                )
                if mol:
                    smiles = Chem.MolToSmiles(mol)
                    img_path = f"png/novel/optimized_{num_nodes}_{i}.png"
                    Draw.MolToImage(mol).save(img_path)
                    optimized_results.append({
                        "SMILES": smiles,
                        "Score": score,
                        "Node_Count": num_nodes,
                        "Image_Path": img_path,
                        "Trajectory": trajectory
                    })
                    print(f"Optimized {i+1}: {smiles} | Score: {score:.4f}")
            except Exception as e:
                print(f"Failed optimization for {num_nodes} nodes, iter {i}: {e}")

    # Save results
    pd.DataFrame(results).to_csv("data/top_scoring_molecules.csv", index=False)
    pd.DataFrame(optimized_results).to_csv("data/optimized_molecule_predictions.csv", index=False)
    print("\nSaved molecule data.")

    # Plot trajectories
    for i, result in enumerate(optimized_results):
        plt.plot(result["Trajectory"], label=f"Molecule {i}")
    plt.xlabel("Optimization Step")
    plt.ylabel("Predicted Binding Score")
    plt.title("Latent Space Optimization Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig("png/novel/optimization_trajectories.png")
    print("Saved optimization trajectory plot.")



if __name__ == "__main__":
    main()
