#!/usr/bin/env python3

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Hyperparameters
HIDDEN_CHANNELS = 256
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT_P = 0.3  # Example dropout probability

def compute_metrics(predictions, targets):
    """Compute additional metrics for regression."""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    mse = np.mean((predictions_np - targets_np) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_np, predictions_np)
    r2 = r2_score(targets_np, predictions_np)
    corr, _ = pearsonr(predictions_np, targets_np)

    return mse, rmse, mae, r2, corr

def visualize_predictions(predictions, targets, title="True vs. Predicted Edge Attributes"):
    """Scatter plot of true vs. predicted values."""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(targets_np, predictions_np, alpha=0.5, label="Data")
    
    # Plot the ideal line
    min_val = min(targets_np.min(), predictions_np.min())
    max_val = max(targets_np.max(), predictions_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    
    plt.xlabel("True Edge Attributes")
    plt.ylabel("Predicted Edge Attributes")
    plt.title(title)
    plt.legend()
    plt.show()

class HeteroGNN(nn.Module):
    def __init__(self, ligand_in_channels=4, target_in_channels=1280,
                 hidden_channels=HIDDEN_CHANNELS, dropout_p=DROPOUT_P):
        super().__init__()
        
        # Define two hetero convolution layers
        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((ligand_in_channels, target_in_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((target_in_channels, ligand_in_channels), hidden_channels)
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('ligand', 'binds_to', 'target'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('target', 'binds_to', 'ligand'): SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        }, aggr='mean')
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)
        
        # Edge predictor (MLP)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        x_dict = {
            'ligand': data['ligand'].x,
            'target': data['target'].x
        }
        edge_index_dict = {
            ('ligand', 'binds_to', 'target'): data['ligand', 'binds_to', 'target'].edge_index,
            ('target', 'binds_to', 'ligand'): data['ligand', 'binds_to', 'target'].edge_index.flip(0)
        }
        
        logger.info(f"Input x_dict: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(self.dropout(x)) for key, x in x_dict.items()}
        logger.info(f"After conv1: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(self.dropout(x)) for key, x in x_dict.items()}
        logger.info(f"After conv2: ligand={x_dict['ligand'].shape}, target={x_dict['target'].shape}")
        
        edge_index = data['ligand', 'binds_to', 'target'].edge_index
        ligand_feats = x_dict['ligand'][edge_index[0]]
        target_feats = x_dict['target'][edge_index[1]]
        edge_feats = torch.cat([ligand_feats, target_feats], dim=-1)
        
        out = self.edge_predictor(edge_feats).squeeze(-1)
        logger.info(f"Edge prediction output: {out.shape}")
        
        return out

def train_model(model, graph, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS):
    """Train the model on train set, validate on val set, then return test set predictions."""
    
    # Split edges into train/val/test
    edge_index = graph['ligand', 'binds_to', 'target'].edge_index
    edge_attr = graph['ligand', 'binds_to', 'target'].edge_attr
    num_edges = edge_index.size(1)

    perm = torch.randperm(num_edges)
    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)
    test_size = num_edges - train_size - val_size

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    # Create train/val/test subgraphs
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

    # Move subgraphs to device
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_graph)
            val_targets = val_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
            val_loss = criterion(val_out, val_targets)

            # Compute additional metrics
            mse, rmse, mae, r2, corr = compute_metrics(val_out, val_targets)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}, "
            f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}"
        )

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_out = model(test_graph)
        test_targets = test_graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
        test_loss = criterion(test_out, test_targets)
        test_mse, test_rmse, test_mae, test_r2, test_corr = compute_metrics(test_out, test_targets)

    logger.info(
        f"=== Final Test Metrics ===\n"
        f"Test Loss (MSE): {test_loss.item():.4f}\n"
        f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, Corr: {test_corr:.4f}"
    )

    return test_out, test_targets

def load_hetero_data(data_path):
    """Load or create normalized graph from a given path."""
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
        # Example: apply log10 transform
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

    # Load data
    graph_path = os.path.join(DATA_DIR, "bindingdb_hetero_graph.pt")
    graph = load_hetero_data(graph_path).to(device)

    # Initialize model
    model = HeteroGNN(
        ligand_in_channels=4,
        target_in_channels=1280,
        hidden_channels=HIDDEN_CHANNELS,
        dropout_p=DROPOUT_P
    ).to(device)

    # Set up training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train model and evaluate on test set
    test_predictions, test_targets = train_model(
        model, graph, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
    )

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(DATA_DIR, "gnn_model.pt"))
    logger.info("Saved trained GNN model to data/gnn_model.pt")

    # (Optional) Visualize predictions on the entire graph or just on the test set
    # Example: visualize predictions on the test set
    visualize_predictions(test_predictions, test_targets, title="Test Set: True vs. Predicted")

    # Example: visualize predictions on the entire dataset
    model.eval()
    with torch.no_grad():
        full_preds = model(graph)
        full_targets = graph['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
    visualize_predictions(full_preds, full_targets, title="Full Dataset: True vs. Predicted")

if __name__ == "__main__":
    main()
