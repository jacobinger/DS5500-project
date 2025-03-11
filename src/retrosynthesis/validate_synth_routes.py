import pandas as pd

def validate_and_rank_routes(scored_routes_file, output_file):
    df = pd.read_csv(scored_routes_file)
    df = df.sort_values(by="gcn_score", ascending=False)  # Higher score = better
    df.to_csv(output_file, index=False)
    print(f"Ranked {len(df)} routes in {output_file}")

if __name__ == "__main__":
    validate_and_rank_routes("results/retrosynth_results/scored_routes.csv",
                            "results/retrosynth_results/ranked_routes.csv")