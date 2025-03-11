import pandas as pd
from utils.molecule_utils import smiles_to_mol
from aizynthfinder.aizynthfinder import AiZynthFinder

def run_retrosynthesis(target_file, zinc_smiles_file, output_dir):
    targets = pd.read_csv(target_file)["smiles"].tolist()
    zinc_smiles = pd.read_csv(zinc_smiles_file)["smiles"].tolist()
    
    finder = AiZynthFinder()
    finder.stock.inchi_from_smiles(zinc_smiles)
    
    routes = []
    for smi in targets[:10]:  # Limit for demo
        mol = smiles_to_mol(smi)
        if mol:
            finder.target_smiles = smi
            finder.tree_search()
            finder.build_routes()
            if finder.routes:
                top_route = finder.routes[0]
                intermediates = [node.smiles for node in top_route["tree"]["children"]]
                routes.append({
                    "target_smiles": smi,
                    "intermediates": intermediates,
                    "route_details": top_route
                })
            else:
                print(f"No route found for {smi}")
    
    df = pd.DataFrame(routes)
    df.to_csv(f"{output_dir}/routes.csv", index=False)
    print(f"Saved {len(routes)} routes to {output_dir}/routes.csv")

if __name__ == "__main__":
    run_retrosynthesis("data/retrosynth/target_molecules.csv",
                      "data/zinc20/zinc20_processed.csv",
                      "data/retrosynth/synthetic_routes")