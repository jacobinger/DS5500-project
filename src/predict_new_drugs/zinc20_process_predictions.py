
import torch
import pandas as pd

from torch.nn.functional import sigmoid

predictions_path = "data/zinc20_predictions.pt"
zinc20_graph_path = "data/zinc20_hetero_graph.pt"

# Load raw logits
predicted_scores = torch.load(predictions_path).cpu()
scaled_scores = sigmoid(predicted_scores).numpy()

zinc20_graph = torch.load(zinc20_graph_path, weights_only=False)
ligand_ids = [f"ZINC20_{i}" for i in range(zinc20_graph['ligand'].num_nodes)]

ranking_df = pd.DataFrame({"Ligand_ID": ligand_ids, "Predicted_Score": scaled_scores})
ranking_df = ranking_df.sort_values(by="Predicted_Score", ascending=False)

top_n = 50
ranking_df = ranking_df.head(top_n)
ranking_df.to_csv("data/zinc20_ranked_candidates.csv", index=False)

print(len(ranking_df))
print(f"Saved top {top_n} ZINC20 candidates to data/zinc20_ranked_candidates.csv")
