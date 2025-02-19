import gzip
from rdkit import Chem
import torch
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt

import gzip
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

sdf_path = 'data/chembl_35.sdf.gz'

# Load the first 5 molecules
mols = []
with gzip.open(sdf_path, 'rb') as f:
    suppl = Chem.ForwardSDMolSupplier(f)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        mols.append(mol)
        if len(mols) == 5:
            break

# Draw the molecules in a grid
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
img.show()  # This will open the default image viewer

# # Alternatively, if you are in a Jupyter Notebook, use:
# # from IPython.display import display
# # display(img)


# # Load external bioactivity data from a CSV file.
# # The CSV should have columns "chembl_id" and "activity"
# bioactivity_file = 'data/chembl_bioactivity.csv'
# bioactivity_df = pd.read_csv(bioactivity_file)
# # Create a dictionary to map chembl_id to activity values.
# bioactivity_dict = pd.Series(bioactivity_df.activity.values, index=bioactivity_df.chembl_id).to_dict()

# def mol_to_graph(mol):
#     """
#     Convert an RDKit molecule to a PyTorch Geometric Data object.
#     Extracts node features and uses the external bioactivity data to set the target.
#     """
#     if mol is None:
#         return None
    
#     # Extract node (atom) features: atomic number and degree.
#     atom_features = []
#     for atom in mol.GetAtoms():
#         atom_features.append([atom.GetAtomicNum(), atom.GetDegree()])
#     x = torch.tensor(atom_features, dtype=torch.float)
    
#     # Build the edge_index (bidirectional edges) from bonds.
#     edge_index = []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         edge_index.append([i, j])
#         edge_index.append([j, i])
#     if edge_index:
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
    
#     data = Data(x=x, edge_index=edge_index)
    
#     # Look up the bioactivity using chembl_id.
#     if mol.HasProp("chembl_id"):
#         chembl_id = mol.GetProp("chembl_id")
#         activity = bioactivity_dict.get(chembl_id, None)
#         if activity is not None:
#             data.y = torch.tensor([activity], dtype=torch.float)
#         else:
#             # Option: skip the molecule or assign a default value.
#             data.y = torch.tensor([0.0], dtype=torch.float)
#     else:
#         data.y = torch.tensor([0.0], dtype=torch.float)
    
#     return data

# def load_dataset(sdf_path, max_mols=None):
#     """
#     Load molecules from a (possibly compressed) SDF file and convert them to graph Data objects.
#     """
#     dataset = []
    
#     if sdf_path.endswith('.gz'):
#         with gzip.open(sdf_path, 'rb') as f:
#             suppl = Chem.ForwardSDMolSupplier(f)
#             for i, mol in enumerate(suppl):
#                 if mol is None:
#                     continue
#                 data = mol_to_graph(mol)
#                 if data is None:
#                     continue
#                 dataset.append(data)
#                 if max_mols and (i + 1) >= max_mols:
#                     break
#     else:
#         suppl = Chem.SDMolSupplier(sdf_path)
#         for i, mol in enumerate(suppl):
#             if mol is None:
#                 continue
#             data = mol_to_graph(mol)
#             if data is None:
#                 continue
#             dataset.append(data)
#             if max_mols and (i + 1) >= max_mols:
#                 break
#     return dataset

# # Load a subset of molecules from the SDF file.
# sdf_file = 'data/chembl_35.sdf.gz'
# dataset = load_dataset(sdf_file, max_mols=1000)
# print(f"Loaded {len(dataset)} molecules from the SDF file.")
