import pandas as pd
from rdkit import Chem

def preprocess_for_retrosynth(input_file, output_file):
    # Load Parkinson’s targets from ChEMBL35
    df = pd.read_csv(input_file)
    smiles_list = df["smiles"].tolist()
    
    # Validate and deduplicate SMILES
    valid_smiles = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.add(smi)
    
    # Save to CSV
    pd.DataFrame({"smiles": list(valid_smiles)}).to_csv(output_file, index=False)
    print(f"Prepared {len(valid_smiles)} target molecules in {output_file}")

if __name__ == "__main__":
    preprocess_for_retrosynth("data/chembl35/pd_targets.csv", "data/retrosynth/target_molecules.csv")