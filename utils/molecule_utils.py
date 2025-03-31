from rdkit import Chem
import torch
from torch_geometric.data import Data
import yaml
import os
from typing import Dict
import logging
import os

def graph_to_molecule(graph):
    mol = Chem.RWMol()
    for atomic_num in graph.x[:, 0].cpu().numpy():
        atom = Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)
    
    max_valence = {6: 4, 7: 3, 8: 2}
    atom_valence = [0] * graph.num_nodes
    for edge in graph.edge_index.t().cpu().numpy():
        i, j = edge
        atom_i, atom_j = int(graph.x[i, 0]), int(graph.x[j, 0])
        if atom_valence[i] < max_valence[atom_i] and atom_valence[j] < max_valence[atom_j]:
            if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                atom_valence[i] += 1
                atom_valence[j] += 1
    
    try:
        Chem.SanitizeMol(mol)
        return mol
    except ValueError as e:
        print(f"Sanitization failed: {e}")
        return None
    
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


def load_config(config_path: str = os.path.join("config", "config.yaml")) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def setup_logger(name: str) -> logging.Logger:
    config = load_config()
    log_level = config.get("logging", {}).get("level", "INFO")
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
