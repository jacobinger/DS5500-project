import pandas as pd
from utils.molecule_utils import is_valid_smiles, canonicalize_smiles

def preprocess_for_retrosynth(input_file, output_file):
    df = pd.read_csv(input_file)
    smiles_list = df["smiles"].tolist()
    
    valid_smiles = set()
    for smi in smiles_list:
        if is_valid_smiles(smi):
            canonical_smi = canonicalize_smiles(smi)
            if canonical_smi:
                valid_smiles.add(canonical_smi)
    
    pd.DataFrame({"smiles": list(valid_smiles)}).to_csv(output_file, index=False)
    print(f"Prepared {len(valid_smiles)} target molecules in {output_file}")

if __name__ == "__main__":
    preprocess_for_retrosynth("data/chembl35/pd_targets.csv", "data/retrosynth/target_molecules.csv")