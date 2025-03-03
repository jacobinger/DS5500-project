#!/usr/bin/env python3

# gnn_model.py
# This script sets up a Graph Neural Network (GNN) model using PyTorch Geometric to predict
# ligand-target interactions (e.g., binding affinities or selectivity) from the BindingDB dataset's
# heterogeneous graph, focusing on polypharmacology and target selectivity for drug discovery.

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, HeteroConv, Linear
from torch_geometric.data import HeteroData
import logging
from torch_geometric.loader import DataLoader
import os
import torch_geometric
from torch_geometric.data.storage import NodeStorage, BaseStorage

torch.serialization.add_safe_globals([NodeStorage, BaseStorage])

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add BaseStorage to safe globals for weights_only=True loading
torch.serialization.add_safe_globals([torch_geometric.data.storage.BaseStorage])

# Directory for data files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Hyperparameters
HIDDEN_SIZE = 128  # Size of hidden layers
NUM_EPOCHS = 50    # Number of training epochs
BATCH_SIZE = 32    # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimization

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean', node_dim=0)
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        # Convert edge_index to tensor if needed
        if isinstance(edge_index, (tuple, list)):
            if all(isinstance(e, torch.Tensor) for e in edge_index):
                edge_index = torch.stack(edge_index, dim=0)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long)

        # In HeteroConv, x is a tuple (x_src, x_dst)
        if isinstance(x, tuple):
            x_src, x_dst = x
        elif isinstance(x, dict):
            # Fallback for dict input (though HeteroConv uses tuples)
            if edge_type == ('ligand', 'binds_to', 'target'):
                x_src = x['ligand']
            elif edge_type == ('target', 'binds_to', 'ligand'):
                x_src = x['target']
            elif edge_type == ('target', 'similar_to', 'target'):
                x_src = x['target']
            else:
                x_src = list(x.values())[0]
        else:
            x_src = x

        # Create edge_attr with correct size
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), x_src.size(-1)),
                                  dtype=x_src.dtype,
                                  device=x_src.device)

        # Propagate expects x to be a tensor or tuple depending on context
        out = self.propagate(edge_index, x=(x_src, x_dst) if isinstance(x, tuple) else x, 
                           edge_attr=edge_attr)
        return self.lin(out)

    def message(self, x_j, edge_attr=None):
        return x_j if edge_attr is None else x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out

# Rest of HeteroGNN class (with minor adjustment)
class HeteroGNN(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_classes=1, 
                 ligand_in_channels=4, target_in_channels=1280):
        super(HeteroGNN, self).__init__()

        self.ligand_lin = Linear(ligand_in_channels, hidden_size)
        self.target_lin = Linear(target_in_channels, hidden_size)

        self.conv1 = HeteroConv({
            ('ligand', 'binds_to', 'target'): GNNLayer(hidden_size, hidden_size),
            ('target', 'binds_to', 'ligand'): GNNLayer(hidden_size, hidden_size),
            # Removed ('target', 'similar_to', 'target') since it's not in your data
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('ligand', 'binds_to', 'target'): GNNLayer(hidden_size, hidden_size),
            ('target', 'binds_to', 'ligand'): GNNLayer(hidden_size, hidden_size),
        }, aggr='mean')

        self.lin1 = Linear(hidden_size * 2, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, data):
        x_dict = {
            'ligand': self.ligand_lin(data['ligand'].x),
            'target': self.target_lin(data['target'].x)
        }

        edge_index_dict = {}
        edge_types = data.edge_types if hasattr(data, 'edge_types') else []

        if ('ligand', 'binds_to', 'target') in edge_types:
            edge_index = self._get_edge_index(data, ('ligand', 'binds_to', 'target'))
            edge_index_dict[('ligand', 'binds_to', 'target')] = edge_index
            edge_index_dict[('target', 'binds_to', 'ligand')] = edge_index.flip(0)
        else:
            logger.warning("Edge type ('ligand', 'binds_to', 'target') not found in data")

        if edge_index_dict:
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: torch.relu(val) for key, val in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            x_dict = {key: torch.relu(val) for key, val in x_dict.items()}
        else:
            logger.warning("No edge types available for message passing")

        out = []
        if ('ligand', 'binds_to', 'target') in edge_types:
            edge_indices = self._get_edge_index(data, ('ligand', 'binds_to', 'target')).t()
            for src, dst in edge_indices:
                ligand_feat = x_dict['ligand'][src]
                target_feat = x_dict['target'][dst]
                combined = torch.cat([ligand_feat, target_feat], dim=-1)
                pred = self.lin2(torch.relu(self.lin1(combined)))
                out.append(pred)

        return torch.stack(out, dim=0) if out else torch.tensor([], device=data['ligand'].x.device)

    def _get_edge_index(self, data, edge_type):
        try:
            if hasattr(data[edge_type], 'edge_index'):
                return data[edge_type].edge_index
            elif hasattr(data[edge_type], 'edge_indices'):
                return data[edge_type].edge_indices
            elif hasattr(data[edge_type], 'adj'):
                return data[edge_type].adj
            else:
                raise AttributeError(f"No edge index found for {edge_type}")
        except Exception as e:
            logger.error(f"Error accessing edge index for {edge_type}: {str(e)}")
            raise
# Update train_model to handle empty outputs
def train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batches_processed = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            
            if out.numel() == 0:
                logger.warning("Model returned empty output for batch")
                continue
                
            if ('ligand', 'binds_to', 'target') in batch.edge_types:
                true_values = batch['ligand', 'binds_to', 'target'].edge_attr.squeeze(-1)
                if len(out) == len(true_values):
                    loss = criterion(out.squeeze(-1), true_values)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches_processed += 1
                else:
                    logger.warning(f"Shape mismatch: out={len(out)}, true_values={len(true_values)}")
            else:
                logger.warning("No 'ligand', 'binds_to', 'target' edge type found in batch")
        
        if batches_processed > 0:
            avg_loss = total_loss / batches_processed
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, No valid batches processed")
            
def load_hetero_data(data_path):
    """
    Load the heterogeneous graph data from a PyTorch file, with weights_only=True for security.
    """
    try:
        logger.info(f"Loading heterogeneous graph from {data_path}")
        graph_path = os.path.join("data", "bindingdb_hetero_graph.pt")
        graph = torch.load(graph_path, weights_only=False)

        # graph = torch.load(data_path, weights_only=False)  # Use weights_only=True for security
        if not isinstance(graph, HeteroData):
            logger.error(f"Loaded data is not a HeteroData object: {type(graph)}")
            raise ValueError("Data must be a HeteroData object")
        return graph
    except Exception as e:
        logger.error(f"Error loading heterogeneous graph: {e}")
        raise

def main():
    """
    Main function to set up and train the GNN model on the BindingDB heterogeneous graph.
    """
    # Load the heterogeneous graph
    graph_path = os.path.join(DATA_DIR, "bindingdb_hetero_graph.pt")
    graph = load_hetero_data(graph_path)

    # Prepare data loader (for batching, if needed)
    # For simplicity, we’ll use the full graph here; for large datasets, split into train/val/test
    train_loader = DataLoader([graph], batch_size=BATCH_SIZE, shuffle=False)  # Adjust for multiple graphs

    # Initialize model (predicting affinity, regression task)
    model = HeteroGNN(hidden_size=HIDDEN_SIZE, num_classes=1)  # 1 for regression (affinity)
    if torch.cuda.is_available():
        model = model.cuda()
        graph = graph.cuda()
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")

    # Define loss function and optimizer (for regression)
    criterion = nn.MSELoss()  # Mean Squared Error for affinity prediction
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(DATA_DIR, "gnn_model.pt"))
    logger.info("Saved trained GNN model to data/gnn_model.pt")

if __name__ == "__main__":
    main()