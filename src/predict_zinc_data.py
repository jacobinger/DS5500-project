
import torch
import os
import logging
from torch_geometric.data import HeteroData
from train_SAGEConv_model import HeteroGNN  # Load the trained model from your existing code

# ✅ Load Trained Model
model_path = "data/gnn_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeteroGNN(ligand_in_channels=1024, target_in_channels=1280).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ✅ Load Processed ZINC20 Graph
zinc20_graph_path = "data/zinc20_hetero_graph.pt"
if not os.path.exists(zinc20_graph_path):
    raise FileNotFoundError(f"ZINC20 graph not found at {zinc20_graph_path}")

zinc20_graph = torch.load(zinc20_graph_path, weights_only=False).to(device)

# ✅ Run Inference on ZINC20
with torch.no_grad():
    predicted_scores = model(zinc20_graph)

# ✅ Save Predictions
predictions_path = "data/zinc20_predictions.pt"
torch.save(predicted_scores, predictions_path)
print(f"✅ Saved ZINC20 predictions at {predictions_path}")
