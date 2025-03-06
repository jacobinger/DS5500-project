#!/usr/bin/env python3

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HIDDEN_CHANNELS = 256
DROPOUT_P = 0.3

class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=1024, target_in_channels=1280,
                 hidden_channels=HIDDEN_CHANNELS, dropout_p=DROPOUT_P):
        super().__init__()
        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((ligand_in_channels, target_in_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((target_in_channels, ligand_in_channels), hidden_channels)
        }, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        }, aggr='mean')
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

def get_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = fp_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)

def load_zinc_smiles(file_path, max_samples=200):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ZINC SMILES file not found at {file_path}. Please download manually from http://files.docking.org/2D/AA/AAAA.smi.gz and place in {DATA_DIR}")
    with open(file_path, "r") as f:
        lines = f.readlines()
        smiles_list = []
        for line in lines[:max_samples]:
            stripped = line.strip()
            if stripped:
                parts = stripped.split()
                if len(parts) >= 1:
                    smiles_list.append(parts[0])
                else:
                    logger.warning(f"Skipping invalid line: {line.strip()}")
        logger.info(f"Loaded {len(smiles_list)} SMILES from {file_path}")
    return smiles_list

def create_prediction_graph(existing_graph, new_ligand_features, target_indices):
    pred_graph = HeteroData()
    pred_graph['target'].x = existing_graph['target'].x
    pred_graph['ligand'].x = torch.tensor(new_ligand_features, dtype=torch.float)
    num_new_ligands = pred_graph['ligand'].x.size(0)
    ligand_idx = torch.arange(num_new_ligands).repeat_interleave(len(target_indices))
    target_idx = target_indices.repeat(num_new_ligands)
    pred_graph['ligand', 'binds_to', 'target'].edge_index = torch.stack([ligand_idx, target_idx], dim=0)
    return pred_graph

def predict_affinities(model, pred_graph, device, mean, std):
    pred_graph = pred_graph.to(device)
    with torch.no_grad():
        predictions = model(pred_graph)
    predictions = (predictions * std) + mean
    return predictions.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} for prediction")

    output_path = os.path.join(DATA_DIR, "chembl_35_hetero_graph.pt")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Preprocessed graph not found at {output_path}")
    graph = torch.load(output_path, weights_only=False)
    logger.info(f"Loaded preprocessed graph from {output_path}")

    edge_attrs = graph["ligand", "binds_to", "target"].edge_attr
    mean = edge_attrs.mean().item()
    std = edge_attrs.std().item()
    graph["ligand", "binds_to", "target"].edge_attr = (edge_attrs - mean) / std
    logger.info(f"Normalized edge attrs: mean={mean}, std={std}")

    model = HeteroGNN(ligand_in_channels=1024, target_in_channels=1280,
                      hidden_channels=HIDDEN_CHANNELS, dropout_p=DROPOUT_P).to(device)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "gnn_model.pt")))
    model.eval()

    zinc_file = os.path.join(DATA_DIR, "zinc_sample.smi")
    new_ligands = load_zinc_smiles(zinc_file, max_samples=200)
    
    valid_ligands = [smi for smi in new_ligands if Chem.MolFromSmiles(smi) is not None]
    new_ligand_features = [get_ligand_features(s) for s in valid_ligands]
    new_ligand_features = [f for f in new_ligand_features if f is not None]
    logger.info(f"Processed {len(new_ligand_features)} valid ligands from ZINC")

    if not new_ligand_features:
        raise ValueError("No valid ligands processed from ZINC file")

    target_counts = torch.bincount(graph["ligand", "binds_to", "target"].edge_index[1])
    top_targets = target_counts.argsort(descending=True)[:10]

    pred_graph = create_prediction_graph(graph, new_ligand_features, top_targets)
    predictions = predict_affinities(model, pred_graph, device, mean, std)

    ligand_ids = [f"LIGAND_{i}" for i in range(len(new_ligand_features))]
    target_ids = [f"CHEMBL{t.item()}" for t in top_targets]
    pairs = [(l, t) for l in ligand_ids for t in target_ids]
    
    results = pd.DataFrame({
        'Ligand': [p[0] for p in pairs],
        'Target': [p[1] for p in pairs],
        'Predicted_pIC50': predictions
    })
    
    results.to_csv("predicted_drug_targets.csv", index=False)
    logger.info("Top 10 predicted drug-target pairs:")
    logger.info(results.sort_values('Predicted_pIC50', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()