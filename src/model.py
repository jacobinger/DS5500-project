import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNTargetPredictor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNNTargetPredictor, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Graph-level pooling
        x = self.lin(x)
        return x

model = GNNTargetPredictor(num_node_features=10, hidden_channels=64, num_classes=1)  # Adjust based on data