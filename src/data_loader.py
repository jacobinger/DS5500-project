from rdkit import Chem
import gzip
import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem

from rdkit import Chem
import gzip

def inspect_sdf_properties(sdf_path):
    """
    Load a compressed SDF file and print properties for the first few molecules.
    """
    with gzip.open(sdf_path, 'rb') as f:
        supplier = Chem.ForwardSDMolSupplier(f)
        for i, mol in enumerate(supplier):
            if mol is not None and i < 5:  # Check the first 5 molecules
                print(f"Molecule {i + 1}:")
                print(f"  SMILES: {Chem.MolToSmiles(mol)}")
                print(f"  Properties: {mol.GetPropsAsDict()}")
            if i >= 4:  # Limit to 5 molecules for brevity
                break

# Specify the path to your SDF file
sdf_path = 'data/chembl_35.sdf.gz'
inspect_sdf_properties(sdf_path)


# def load_sdf_gz(sdf_path):
#     """
#     Load and parse a compressed SDF (.sdf.gz) file using RDKit.
#     Returns a list of RDKit molecule objects.
#     """
#     molecules = []
#     try:
#         with gzip.open(sdf_path, 'rb') as f:
#             supplier = Chem.ForwardSDMolSupplier(f)
#             for mol in supplier:
#                 if mol is not None:  # Skip invalid molecules
#                     molecules.append(mol)
#         print(f"Loaded {len(molecules)} molecules from {sdf_path}")
#     except Exception as e:
#         print(f"Error loading SDF file: {e}")
#     return molecules

# # Specify the path to your SDF file
# sdf_path = 'data/chembl_35.sdf.gz'
# molecules = load_sdf_gz(sdf_path)



# def mol_to_graph(mol):
#     """
#     Convert an RDKit molecule to a PyTorch Geometric graph.
#     """
#     # Get atoms (nodes)
#     num_atoms = mol.GetNumAtoms()
#     atom_features = []
#     for atom in mol.GetAtoms():
#         # Example features: atomic number, degree, hybridization, etc.
#         features = [
#             atom.GetAtomicNum(),  # Atomic number
#             atom.GetDegree(),     # Number of bonds
#             atom.GetHybridization().real if atom.GetHybridization() else 0,  # Hybridization
#             atom.GetFormalCharge(),  # Charge
#         ]
#         atom_features.append(features)
#     x = torch.tensor(atom_features, dtype=torch.float)  # Node features

#     # Get bonds (edges)
#     edge_index = []
#     edge_attr = []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         edge_index.append([i, j])
#         edge_index.append([j, i])  # Undirected graph: add reverse edge
#         # Bond type as edge feature (e.g., 1 for single, 2 for double, 3 for triple)
#         bond_type = bond.GetBondTypeAsDouble()
#         edge_attr.append([bond_type])
#         edge_attr.append([bond_type])
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Shape: [2, num_edges]
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Edge features

#     # Create graph data object
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#     # Optionally add graph-level labels or properties (e.g., from SDF tags)
#     # Example: Extract a property like activity or target ID if available
#     props = mol.GetPropsAsDict()
#     if 'ACTIVITY' in props:  # Adjust based on actual property names in your SDF
#         data.y = torch.tensor([float(props['ACTIVITY'])], dtype=torch.float)  # Example label

#     return data

# # Convert all molecules to graphs
# graph_dataset = [mol_to_graph(mol) for mol in molecules]

# import pandas as pd

# metadata = pd.read_csv('data/metadata.csv')  # Adjust path and format
# # Example: Match compound ID from SDF to metadata to add labels or features
# for graph, mol in zip(graph_dataset, molecules):
#     mol_id = mol.GetProp('_Name')  # Or another identifier in the SDF
#     if mol_id in metadata['compound_id'].values:
#         row = metadata[metadata['compound_id'] == mol_id]
#         graph.y = torch.tensor([float(row['activity'])], dtype=torch.float)  # Example