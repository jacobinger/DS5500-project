import pandas as pd
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from utils.molecule_utils import smiles_to_graph

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)  # Global pooling

def train_gcn(zinc_file, output_model):
    # Load ZINC20 data (assume feasibility = 1 for simplicity)
    df = pd.read_csv(zinc_file)
    graphs = [smiles_to_graph(smi) for smi in df["smiles"] if smiles_to_graph(smi)]
    labels = torch.ones(len(graphs), 1)  # Dummy labels (all feasible)
    
    # DataLoader
    dataset = [g for g in graphs if g]
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(10):  # Limited epochs for demo
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, labels[:len(batch)])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), output_model)
    print(f"Saved GCN model to {output_model}")

if __name__ == "__main__":
    train_gcn("data/zinc20/zinc20_processed.csv", "models/gcn_model.pth")