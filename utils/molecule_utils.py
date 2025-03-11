from rdkit import Chem
import torch
from torch_geometric.data import Data

def smiles_to_mol(smiles):
    """Convert SMILES to RDKit molecule, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol else None
    except:
        return None

def is_valid_smiles(smiles):
    """Check if SMILES is valid."""
    return bool(smiles_to_mol(smiles))

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form, return None if invalid."""
    mol = smiles_to_mol(smiles)
    return Chem.MolToSmiles(mol) if mol else None

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric graph."""
    mol = smiles_to_mol(smiles)
    if not mol:
        return None
    
    # Node features (atom type, degree)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum(), atom.GetDegree()])
    
    # Edge indices (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected graph
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)