import pandas as pd

def validate_and_rank_routes(routes_file, output_file):
    df = pd.read_csv(routes_file)
    
    # Add simple ranking metric: number of intermediates (fewer steps = better)
    df["num_steps"] = df["intermediates"].apply(lambda x: len(eval(x)))
    df = df.sort_values(by="num_steps", ascending=True)  # Rank by shortest routes
    
    # Save ranked routes
    df.to_csv(output_file, index=False)
    print(f"Ranked {len(df)} routes in {output_file}")

if __name__ == "__main__":
    validate_and_rank_routes("data/retrosynth/synthetic_routes/routes.csv",
                            "results/retrosynth_results/ranked_routes.csv")