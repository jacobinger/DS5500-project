#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HIDDEN_CHANNELS = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=4, target_in_channels=1280, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((ligand_in_channels, target_in_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((target_in_channels, ligand_in_channels), hidden_channels)
        }, aggr='mean')
        # Linear layer for edge prediction: concatenate ligand and target features (2 * hidden_channels)
        self.edge_predictor = nn.Linear(2 * hidden_channels, 1)
    
    def forward(self, data):
        x_dict = {'ligand': data['ligand'].x, 'target': data['target'].x}
        edge_index_dict = {
            ('ligand', 'binds_to', 'target'): data['ligand', 'binds_to', 'target'].edge_index,
            ('target', 'binds_to', 'ligand'): data['ligand', 'binds_to', 'target'].edge_index.flip(0)
        }
        
        logger.info(f"Input x_dict: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        
        # Update node features
        orig_x_dict = x_dict.copy()
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['ligand'] = x_dict.get('ligand', orig_x_dict['ligand'])
        x_dict['target'] = x_dict.get('target', orig_x_dict['target'])
        logger.info(f"After conv1: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        logger.info(f"After ReLU: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        
        # Edge prediction
        edge_index = data['ligand', 'binds_to', 'target'].edge_index
        ligand_feats = x_dict['ligand'][edge_index[0]]  # [6362, 128]
        target_feats = x_dict['target'][edge_index[1]]  # [6362, 128]
        edge_feats = torch.cat([ligand_feats, target_feats], dim=-1)  # [6362, 256]
        out = self.edge_predictor(edge_feats).squeeze(-1)  # [6362]
        
        logger.info(f"Edge prediction output: {out.shape}")
        return out

def train_model(model, graph, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph)
        targets = graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
        
        if out.numel() == 0:
            logger.warning("Model returned empty output")
            continue
        
        loss = criterion(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def load_hetero_data(data_path):
    normalized_path = os.path.join(DATA_DIR, "bindingdb_hetero_graph_normalized.pt")
    
    if os.path.exists(normalized_path):
        logger.info(f"Loading normalized graph from {normalized_path}")
        graph = torch.load(normalized_path, weights_only=False)
    else:
        logger.info(f"Loading graph from {data_path} and normalizing")
        graph = torch.load(data_path, weights_only=False)
        if 'ligand' not in graph or 'target' not in graph:
            raise ValueError("Graph missing 'ligand' or 'target' node types")
        edge_attr = graph['ligand', 'binds_to', 'target'].edge_attr
        edge_attr = torch.log10(edge_attr + 1)
        graph['ligand', 'binds_to', 'target'].edge_attr = edge_attr.clone()
        logger.info(f"Normalized edge attrs: {edge_attr.min():.4f} to {edge_attr.max():.4f}")
        torch.save(graph, normalized_path)
    
    if not isinstance(graph, HeteroData):
        raise ValueError(f"Loaded data is not a HeteroData object: {type(graph)}")
    if graph['ligand'].x is None or graph['target'].x is None:
        raise ValueError("Graph missing node features: ligand.x or target.x is None")
    logger.info(f"Loaded graph: {graph}")
    return graph

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} for training")

    graph_path = os.path.join(DATA_DIR, "bindingdb_hetero_graph.pt")
    graph = load_hetero_data(graph_path).to(device)

    model = HeteroGNN(
        ligand_in_channels=4,
        target_in_channels=1280,
        hidden_channels=HIDDEN_CHANNELS
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, graph, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "gnn_model.pt"))
    logger.info("Saved trained GNN model to data/gnn_model.pt")

if __name__ == "__main__":
    main()