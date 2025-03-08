#!/usr/bin/env python3

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
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
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT_P = 0.2

def compute_metrics(predictions, targets):
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    mse = np.mean((predictions_np - targets_np) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_np, predictions_np)
    r2 = r2_score(targets_np, predictions_np)
    corr, _ = pearsonr(predictions_np, targets_np)
    return mse, rmse, mae, r2, corr

def visualize_predictions(predictions, targets, title="True vs. Predicted Edge Attributes", mean=0, std=1):
    mean = mean.item()  # Convert tensor to scalar
    std = std.item()    # Convert tensor to scalar
    predictions_np = (predictions.detach().cpu().numpy() * std) + mean
    targets_np = (targets.detach().cpu().numpy() * std) + mean
    plt.figure(figsize=(8, 6))
    plt.scatter(targets_np, predictions_np, alpha=0.5, label="Data")
    min_val = min(targets_np.min(), predictions_np.min())
    max_val = max(targets_np.max(), predictions_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    plt.xlabel("True Edge Attributes")
    plt.ylabel("Predicted Edge Attributes")
    plt.title(title)
    plt.legend()
    plt.show()

class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=1024, target_in_channels=1280,
                 hidden_channels=HIDDEN_CHANNELS, dropout_p=DROPOUT_P):
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

def train_model(model, graph, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, seed=42):
    torch.manual_seed(seed)
    edge_index = graph['ligand', 'binds_to', 'target'].edge_index
    edge_attr = graph['ligand', 'binds_to', 'target'].edge_attr
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    train_graph = HeteroData()
    train_graph['ligand'].x = graph['ligand'].x
    train_graph['target'].x = graph['target'].x
    train_graph['ligand', 'binds_to', 'target'].edge_index = edge_index[:, train_idx]
    train_graph['ligand', 'binds_to', 'target'].edge_attr = edge_attr[train_idx]

    val_graph = HeteroData()
    val_graph['ligand'].x = graph['ligand'].x
    val_graph['target'].x = graph['target'].x
    val_graph['ligand', 'binds_to', 'target'].edge_index = edge_index[:, val_idx]
    val_graph['ligand', 'binds_to', 'target'].edge_attr = edge_attr[val_idx]

    test_graph = HeteroData()
    test_graph['ligand'].x = graph['ligand'].x
    test_graph['target'].x = graph['target'].x
    test_graph['ligand', 'binds_to', 'target'].edge_index = edge_index[:, test_idx]
    test_graph['ligand', 'binds_to', 'target'].edge_attr = edge_attr[test_idx]

    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)
    test_graph = test_graph.to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph)
        targets = train_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
        loss = criterion(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_graph)
            val_targets = val_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
            val_loss = criterion(val_out, val_targets)
            mse, rmse, mae, r2, corr = compute_metrics(val_out, val_targets)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                    f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}")

    model.eval()
    with torch.no_grad():
        test_out = model(test_graph)
        test_targets = test_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
        test_loss = criterion(test_out, test_targets)
        test_mse, test_rmse, test_mae, test_r2, test_corr = compute_metrics(test_out, test_targets)

    logger.info(f"=== Final Test Metrics ===\nTest Loss: {test_loss.item():.4f}\n"
                f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, Corr: {test_corr:.4f}")
    return test_out, test_targets, test_targets.mean(), test_targets.std()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} for training")

    output_path = os.path.join(DATA_DIR, "chembl_35_hetero_graph_PD.pt")
    if os.path.exists(output_path):
        graph = torch.load(output_path, weights_only=False)
        logger.info(f"Loaded preprocessed graph from {output_path}")
        logger.info(f"Graph: {graph}")
        edge_attrs = graph["ligand", "binds_to", "target"].edge_attr
        mean, std = edge_attrs.mean(), edge_attrs.std()
        #Normalize Data: Add normalization to edge_attr (affinities) for training stability
        graph["ligand", "binds_to", "target"].edge_attr = (edge_attrs - mean) / std
        print(f"Number of nodes (ligand): {graph['ligand'].x.shape[0]}")
        print(f"Number of nodes (target): {graph['target'].x.shape[0]}")
        print(f"Number of edges: {graph['ligand', 'binds_to', 'target'].edge_index.shape[1]}")


        num_nodes = graph['ligand'].x.shape[0] + graph['target'].x.shape[0]
        num_edges = graph['ligand', 'binds_to', 'target'].edge_index.shape[1]
        density = num_edges / num_nodes
        print(f"Graph Density: {density:.4f}")

        logger.info(f"Normalized edge attrs: mean={mean}, std={std}")
        logger.info(f"Edge attr stats: min={edge_attrs.min()}, max={edge_attrs.max()}, mean={mean}, std={std}")
    else:
        raise FileNotFoundError(f"Preprocessed graph not found at {output_path}. Please run preprocessing first.")

    graph = graph.to(device)
    model = HeteroGNN(
        ligand_in_channels=1024,
        target_in_channels=1280,
        hidden_channels=HIDDEN_CHANNELS,
        dropout_p=DROPOUT_P
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    test_predictions, test_targets, edge_mean, edge_std = train_model(
        model, graph, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, seed=42
    )
    torch.save(model.state_dict(), os.path.join(DATA_DIR, "gnn_model.pt"))
    logger.info("Saved trained GNN model to data/gnn_model.pt")
    visualize_predictions(test_predictions, test_targets, title="Test Set: True vs. Predicted", mean=edge_mean, std=edge_std)

if __name__ == "__main__":
    main()