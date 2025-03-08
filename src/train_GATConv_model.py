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
from torch_geometric.nn import HeteroConv, GATConv
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
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_P = 0.3

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


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData

class ImprovedHeteroGAT(nn.Module):
    def __init__(self, ligand_in_channels=1024, target_in_channels=1280,
                 hidden_channels=512, heads=8, dropout_p=0.5):
        super().__init__()

        # First GAT Layer (More Heads)
        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): GATConv((ligand_in_channels, target_in_channels), 
                                                      hidden_channels, 
                                                      heads=heads, 
                                                      dropout=dropout_p, 
                                                      add_self_loops=False),
            ('target', 'binds_to', 'ligand'): GATConv((target_in_channels, ligand_in_channels), 
                                                      hidden_channels, 
                                                      heads=heads, 
                                                      dropout=dropout_p, 
                                                      add_self_loops=False)
        }, aggr='mean')

        self.norm1 = nn.LayerNorm(hidden_channels * heads)  # 🚀 Use LayerNorm

        # Second GAT Layer
        self.conv2 = HeteroConv({
            ('ligand', 'binds_to', 'target'): GATConv((hidden_channels * heads, hidden_channels * heads), 
                                                      hidden_channels, 
                                                      heads=heads, 
                                                      dropout=dropout_p, 
                                                      add_self_loops=False),
            ('target', 'binds_to', 'ligand'): GATConv((hidden_channels * heads, hidden_channels * heads), 
                                                      hidden_channels, 
                                                      heads=heads, 
                                                      dropout=dropout_p, 
                                                      add_self_loops=False)
        }, aggr='mean')

        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.dropout = nn.Dropout(dropout_p)

        # Edge Prediction (Stronger MLP)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels * heads, hidden_channels),
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

        # Apply GAT layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.norm1(torch.relu(self.dropout(x))) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.norm2(torch.relu(self.dropout(x))) for key, x in x_dict.items()}

        # Edge Predictions
        edge_index = data['ligand', 'binds_to', 'target'].edge_index
        ligand_feats = x_dict['ligand'][edge_index[0]]
        target_feats = x_dict['target'][edge_index[1]]
        edge_feats = torch.cat([ligand_feats, target_feats], dim=-1)

        return self.edge_predictor(edge_feats).squeeze(-1)


def train_model(model, graph, device, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.HuberLoss(delta=1.0)  # More robust loss function

    train_graph = graph.to(device)

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(train_graph)  # Placeholder for validation data
            val_loss = criterion(val_out, targets)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    print("Training Complete!")

    with torch.no_grad():
        test_out = model(train_graph)  # Placeholder for test data
        test_targets = train_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
        test_mean, test_std = test_targets.mean(), test_targets.std()

    return test_out, test_targets, test_mean, test_std


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
        logger.info(f"Normalized edge attrs: mean={mean}, std={std}")
        logger.info(f"Edge attr stats: min={edge_attrs.min()}, max={edge_attrs.max()}, mean={mean}, std={std}")
    else:
        raise FileNotFoundError(f"Preprocessed graph not found at {output_path}. Please run preprocessing first.")

    graph = graph.to(device)
    model = ImprovedHeteroGAT(
        ligand_in_channels=1024,
        target_in_channels=1280,
        hidden_channels=HIDDEN_CHANNELS,
        dropout_p=DROPOUT_P
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    test_predictions, test_targets, edge_mean, edge_std = train_model(
        model, graph, device, num_epochs=100  
    )

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "gnn_model.pt"))
    logger.info("Saved trained GNN model to data/gnn_model.pt")
    visualize_predictions(test_predictions, test_targets, title="Test Set: True vs. Predicted", mean=edge_mean, std=edge_std)

if __name__ == "__main__":
    main()