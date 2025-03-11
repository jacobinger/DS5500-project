import pandas as pd
from rdkit import Chem
# Note: Replace 'aizynthfinder' with your chosen tool (e.g., ASKCOS requires different setup)
from aizynthfinder.aizynthfinder import AiZynthFinder
#watch video bellow
# https://github.com/MolecularAI/aizynthfinder

def run_retrosynthesis(target_file, zinc_smiles_file, output_dir):
    # Load data
    targets = pd.read_csv(target_file)["smiles"].tolist()
    zinc_smiles = pd.read_csv(zinc_smiles_file)["smiles"].tolist()
    
    # Initialize retrosynthesis tool (example with AiZynthFinder)
    finder = AiZynthFinder()
    finder.stock.inchi_from_smiles(zinc_smiles)  # Use ZINC20 as precursor stock
    
    routes = []
    for smi in targets[:10]:  # Limit for demo; remove for full run
        mol = Chem.MolFromSmiles(smi)
        if mol:
            finder.target_smiles = smi
            finder.tree_search()  # Generate synthetic routes
            finder.build_routes()
            if finder.routes:
                # Extract intermediates from the top route
                top_route = finder.routes[0]
                intermediates = [node.smiles for node in top_route["tree"]["children"]]
                routes.append({
                    "target_smiles": smi,
                    "intermediates": intermediates,
                    "route_details": top_route  # Full route for debugging
                })
            else:
                print(f"No route found for {smi}")
    
    # Save results
    df = pd.DataFrame(routes)
    df.to_csv(f"{output_dir}/routes.csv", index=False)
    print(f"Saved {len(routes)} routes to {output_dir}/routes.csv")

if __name__ == "__main__":
    run_retrosynthesis("data/retrosynth/target_molecules.csv",
                      "data/zinc20/zinc20_processed.csv",
                      "data/retrosynth/synthetic_routes")