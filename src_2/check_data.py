#!/usr/bin/env python3

import os
import sqlite3
import logging
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load ESM-2 for target embeddings (1280D)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

def get_ligand_features(smiles):
    """Generate 4D features for ligands (e.g., MW, LogP, HBA, HBD)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ])

def get_target_embedding(sequence):
    """Generate 1280D ESM-2 embedding for protein sequence."""
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool over sequence length
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def process_chembl_35(db_path, output_path, max_pairs=10000):
    """Process ChEMBL 35 into HeteroData."""
    conn = sqlite3.connect(db_path)
    print(conn)
    
    # Query bioactivity data (ligand-target pairs with affinities)
    query = """
        SELECT DISTINCT
            md.chembl_id AS ligand_id,
            cs.canonical_smiles AS smiles,
            td.chembl_id AS target_id,
            csq.sequence AS target_sequence,
            act.standard_value AS affinity,
            act.standard_type AS affinity_type,
            act.standard_units AS affinity_units
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN activities act ON md.molregno = act.molregno
        JOIN target_dictionary td ON act.target_id = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences csq ON tc.component_id = csq.component_id
        WHERE act.standard_value IS NOT NULL
            AND act.standard_type IN ('IC50', 'Kd', 'Ki')
            AND act.standard_units = 'nM'
            AND cs.canonical_smiles IS NOT NULL
            AND csq.sequence IS NOT NULL
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(max_pairs,))
    conn.close()

    logger.info(f"Extracted {len(df)} ligand-target pairs")

    # Initialize HeteroData
    graph = HeteroData()
    ligand_dict = {}
    target_dict = {}
    edge_list = []
    edge_attrs = []

    # Process ligands and targets
    for idx, row in df.iterrows():
        ligand_id = row["ligand_id"]
        target_id = row["target_id"]

        # Ligand features
        if ligand_id not in ligand_dict:
            feats = get_ligand_features(row["smiles"])
            if feats is not None:
                ligand_dict[ligand_id] = (len(ligand_dict), feats)
        
        # Target features
        if target_id not in target_dict:
            embedding = get_target_embedding(row["target_sequence"])
            target_dict[target_id] = (len(target_dict), embedding)

        # Edges and attributes
        if ligand_id in ligand_dict and target_id in target_dict:
            ligand_idx = ligand_dict[ligand_id][0]
            target_idx = target_dict[target_id][0]
            affinity = -np.log10(row["affinity"] / 1e9 + 1e-10)  # pIC50/pKd, avoid log(0)
            edge_list.append([ligand_idx, target_idx])
            edge_attrs.append(affinity)

    # Populate HeteroData
    graph["ligand"].x = torch.tensor(
        [val[1] for val in ligand_dict.values()], dtype=torch.float
    )
    graph["target"].x = torch.tensor(
        [val[1] for val in target_dict.values()], dtype=torch.float
    )
    graph["ligand", "binds_to", "target"].edge_index = torch.tensor(
        edge_list, dtype=torch.long
    ).t().contiguous()
    graph["ligand", "binds_to", "target"].edge_attr = torch.tensor(
        edge_attrs, dtype=torch.float
    ).unsqueeze(-1)

    logger.info(f"Graph: {graph}")
    torch.save(graph, output_path)
    return graph

def main():
    db_path = os.path.join(DATA_DIR, "chembl_35.db")
    print('db_path:', db_path)
    output_path = os.path.join(DATA_DIR, "chembl_35_hetero_graph.pt")
    
    # Process ChEMBL 35
    graph = process_chembl_35(db_path, output_path, max_pairs=10000)
    
    # Update your existing load_hetero_data to use this
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.to(device)
    
    # Your existing model training code here...

if __name__ == "__main__":
    main()