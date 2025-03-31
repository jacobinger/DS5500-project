import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from rdkit.Chem import AllChem

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
from create_new_drugs.VGAE_affinity_utils import generate_n_molecules

output_dir = "png/novel"
os.makedirs(output_dir, exist_ok=True)


# ensure reproducibility by setting rabndom seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=1024, target_in_channels=1280,
                 hidden_channels=256, dropout_p=0.2):
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
        # data is a HeteroData object with 'ligand' and 'target' nodes.
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

def molecule_to_ligand_features(mol, radius=2, n_bits=1024):
    """
    Convert an RDKit Mol object into Morgan fingerprint features.
    Returns a [1, n_bits] torch.FloatTensor.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr).unsqueeze(0)

def prepare_evaluation_data(mol, target_features):
    """
    Prepare a HeteroData object for evaluating a generated molecule.
    
    Parameters:
      mol: an RDKit Mol object (the generated molecule).
      target_features: a torch.Tensor of shape [1, 1280] representing the target.
      
    Returns:
      A HeteroData object containing ligand and target nodes and a single edge.
    """
    from torch_geometric.data import HeteroData
    ligand_features = molecule_to_ligand_features(mol)  # shape [1, 1024]
    data = HeteroData()
    data['ligand'].x = ligand_features
    data['target'].x = target_features  # shape [1, 1280]
    # Create an edge from the single ligand (node 0) to the single target (node 0).
    data['ligand', 'binds_to', 'target'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    return data

def evaluate_generated_molecule(mol, gnn_model, device, target_features, threshold=0.5, use_sigmoid=True):
    """
    Evaluate a generated molecule using the hetero GNN model on the same scale as ZINC20 predictions.

    Parameters:
      mol: RDKit Mol object.
      gnn_model: Trained hetero GNN model.
      device: torch.device.
      target_features: torch.Tensor of shape [1, 1280].
      threshold: Threshold to decide viability.
      use_sigmoid: Whether to apply sigmoid scaling to match predictions.

    Returns:
      score: Predicted binding affinity (float).
      is_viable: Boolean.
    """
    gnn_model.eval()
    data = prepare_evaluation_data(mol, target_features)
    data = data.to(device)

    with torch.no_grad():
        output = gnn_model(data)
        score_tensor = output if isinstance(output, torch.Tensor) else output['score']
        if use_sigmoid:
            score = torch.sigmoid(score_tensor).item()
        else:
            score = score_tensor.item()

    is_viable = score > threshold
    return score, is_viable


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define a range of node counts to explore.
    node_counts = [5, 6, 7, 8, 9]  # You can adjust this range based on chemical intuition.
    molecules_per_count = 3        # Number of molecules to generate per node count.
    
    # A dictionary to store generated molecules for each node count.
    all_generated_molecules = {}
    for num_nodes in node_counts:
        print(f"\nGenerating molecules with {num_nodes} nodes:")
        generated_molecules = generate_n_molecules(n=molecules_per_count, num_nodes=num_nodes)
        all_generated_molecules[num_nodes] = generated_molecules
    
    # Load the pre-trained GNN model for evaluation.
    model_path = os.path.join("data", "sageconv_model.pt")
    gnn_model = HeteroGNN(ligand_in_channels=1024, target_in_channels=1280,
                           hidden_channels=256, dropout_p=0.2).to(device)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    
    target_features = torch.rand(1, 1280, device=device)
    
    # Ensure the output directory exists.
    output_dir = os.path.join("png", "novel")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each generated molecule.
    for num_nodes, molecules in all_generated_molecules.items():
        print(f"\nEvaluating molecules with {num_nodes} nodes:")
        for i, mol in enumerate(molecules):
            print(f"\nMolecule {i+1} (nodes: {num_nodes}):")
            if mol:
                smiles = Chem.MolToSmiles(mol)
                print("SMILES:", smiles)
                score, viable = evaluate_generated_molecule(mol, gnn_model, device, target_features, threshold=0.5)
                print(f"Predicted binding affinity score: {score:.4f}")
                if viable:
                    print("=> Molecule is predicted to be a good binder (viable/matchable).")
                else:
                    print("=> Molecule is NOT predicted to be a good binder.")
                img = Draw.MolToImage(mol)
                image_path = os.path.join(output_dir, f"generated_molecule_{num_nodes}_nodes_{i}.png")
                img.save(image_path)
                print(f"Saved image to {image_path}")
            else:
                print("Failed to generate a valid molecule.")

if __name__ == "__main__":
    main()
