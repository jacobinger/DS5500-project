import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_affinity_scores(csv_path, output_path="plots/affinity_plot.png", top_n=None, show_viable=True):
    """
    Plots binding affinity scores from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file with columns 'SMILES', 'Score', 'Viable'.
        output_path (str): Path to save the plot.
        top_n (int): Limit to top N rows by score.
        show_viable (bool): Color viable and non-viable molecules differently.
    """
    df = pd.read_csv(csv_path)
    
    if top_n is not None:
        df = df.sort_values(by="Score", ascending=False).head(top_n)
    
    smiles_labels = df['SMILES'].fillna("Invalid")
    scores = df['Score']
    
    plt.figure(figsize=(12, 6))
    
    if show_viable and 'Viable' in df.columns:
        viable_mask = df['Viable']
        plt.bar(range(len(df)), scores, color=viable_mask.map({True: "green", False: "red"}))
    else:
        plt.bar(range(len(df)), scores, color="blue")

    plt.xticks(range(len(df)), smiles_labels, rotation=90, fontsize=6)
    plt.ylabel("Binding Affinity Score")
    plt.title("Predicted Binding Affinities of Generated Molecules")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved affinity plot to {output_path}")
    plt.close()
