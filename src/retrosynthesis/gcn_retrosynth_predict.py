import pandas as pd
import torch
from torch_geometric.nn import GCNConv
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
        return x.mean(dim=0)

def predict_with_gcn(routes_file, model_path, output_file):
    model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    df = pd.read_csv(routes_file)
    scores = []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            intermediates = eval(row["intermediates"])
            route_score = 0
            for smi in intermediates:
                graph = smiles_to_graph(smi)
                if graph:
                    score = model(graph).item()  # Feasibility score
                    route_score += score
            avg_score = route_score / len(intermediates) if intermediates else 0
            scores.append(avg_score)
    
    df["gcn_score"] = scores
    df.to_csv(output_file, index=False)
    print(f"Saved scored routes to {output_file}")

if __name__ == "__main__":
    predict_with_gcn("data/retrosynth/synthetic_routes/routes.csv",
                     "models/gcn_model.pth",
                     "results/retrosynth_results/scored_routes.csv")