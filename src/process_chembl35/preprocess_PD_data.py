import os
import sqlite3
import logging
import torch
import numpy as np
import random
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load ESM-2
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model.eval()

def get_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1280)
    fp = fp_gen.GetFingerprint(mol)
    # Convert the fingerprint to a list of bits, then to a numpy array
    arr = np.array(list(fp), dtype=np.float32)
    return arr

def get_target_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# NEW: Fetch additional PD targets based on EFO/MONDO annotations
def fetch_disease_linked_pd_targets(conn):
    query = """
    SELECT DISTINCT td.chembl_id
    FROM drug_indication di
    JOIN molecule_dictionary md ON di.molregno = md.molregno
    JOIN mechanism m ON md.molregno = m.molregno
    JOIN target_dictionary td ON m.tid = td.tid
    WHERE di.efo_id IN (
        'EFO_0002508',  -- Parkinson's
        'MONDO_0005180',
        'EFO_0002506',  -- Huntington's
        'EFO_0000249'   -- Alzheimer's
    )
      AND td.organism = 'Homo sapiens'
    """
    return pd.read_sql_query(query, conn)['chembl_id'].tolist()


# NEW: Merge curated + disease-derived target list
def get_expanded_pd_target_ids(conn):
    manual_ids = [
        'CHEMBL1795186', 'CHEMBL6151', 'CHEMBL2176839', 'CHEMBL5408', 'CHEMBL6122',
        'CHEMBL2056', 'CHEMBL1075104', 'CHEMBL2782', 'CHEMBL1663', 'CHEMBL1937',
        'CHEMBL217', 'CHEMBL234', 'CHEMBL251', 'CHEMBL224', 'CHEMBL216', 'CHEMBL3227',
        'CHEMBL1075104', 'CHEMBL3337330',
        'CHEMBL1862', 'CHEMBL2828',
        'CHEMBL2039', 'CHEMBL2023', 'CHEMBL1843', 'CHEMBL2179', 'CHEMBL6159',
        'CHEMBL5169188', 'CHEMBL220', 'CHEMBL238', 'CHEMBL4138', 'CHEMBL5183',
        'CHEMBL6152'
    ]
    try:
        disease_ids = fetch_disease_linked_pd_targets(conn)
        combined_ids = sorted(set(manual_ids + disease_ids))
        logger.info(f"Expanded PD target list using disease annotations: {len(combined_ids)} targets")
        return combined_ids
    except Exception as e:
        logger.warning(f"Falling back to manual PD targets due to: {e}")
        return manual_ids

# UPDATED: Expanded PD query using dynamic target set
def extract_pd_data(conn, pd_target_ids):
    placeholders = ', '.join(f"'{tid}'" for tid in pd_target_ids)
    query = f"""
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
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences csq ON tc.component_id = csq.component_id
    WHERE act.standard_value IS NOT NULL
        AND act.standard_type IN (
            'IC50', 'Ki', 'Kd', 'EC50', 'Potency', 'AC50', 'EC90', 'pIC50', 'pKi',
            'Inhibition', 'Activity', 'Percent inhibition', 'Binding'
        )
        AND act.standard_units IN ('nM', 'uM', 'µM', 'pIC50', 'log(M)')
        AND cs.canonical_smiles IS NOT NULL
        AND csq.sequence IS NOT NULL
        AND td.chembl_id IN ({placeholders})
        AND a.confidence_score >= 5
    ORDER BY act.activity_id;
    """
    return pd.read_sql_query(query, conn)

def build_graph_from_df(df, ligand_dict, target_dict):
    graph = HeteroData()
    edge_list = []
    edge_attrs = []

    for idx, row in df.iterrows():
        ligand_id = row["ligand_id"]
        target_id = row["target_id"]
        if ligand_id in ligand_dict and target_id in target_dict:
            if row["affinity"] <= 0:
                continue
            ligand_idx = ligand_dict[ligand_id][0]
            target_idx = target_dict[target_id][0]
            affinity = -np.log10(row["affinity"] / 1e9 + 1e-10)
            edge_list.append([ligand_idx, target_idx])
            edge_attrs.append(affinity)

    ligand_features = np.array([val[1] for val in ligand_dict.values()])
    target_features = np.array([val[1] for val in target_dict.values()])
    graph["ligand"].x = torch.tensor(ligand_features, dtype=torch.float)
    graph["target"].x = torch.tensor(target_features, dtype=torch.float)
    graph["ligand", "binds_to", "target"].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    graph["ligand", "binds_to", "target"].edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(-1)
    return graph

def process_chembl_35(db_path):
    conn = sqlite3.connect(db_path)
    pd_target_ids = get_expanded_pd_target_ids(conn)
    logger.info(f"Using expanded list with {len(pd_target_ids)} PD target IDs.")

    df = extract_pd_data(conn, pd_target_ids)
    conn.close()
    logger.info(f"Extracted {len(df)} ligand-target rows related to PD.")

    ligand_dict = {}
    target_dict = {}

    for idx, row in df.iterrows():
        lid, tid = row["ligand_id"], row["target_id"]
        if lid not in ligand_dict:
            feats = get_ligand_features(row["smiles"])
            if feats is not None:
                ligand_dict[lid] = (len(ligand_dict), feats)
        if tid not in target_dict:
            embedding = get_target_embedding(row["target_sequence"])
            target_dict[tid] = (len(target_dict), embedding)

    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    logger.info(f"Train 1 samples: {len(train_df)}, Train 2 samples: {len(test_df)}")

    train_graph = build_graph_from_df(train_df, ligand_dict, target_dict)
    test_graph = build_graph_from_df(test_df, ligand_dict, target_dict)

    torch.save(train_graph, os.path.join(DATA_DIR, "train_1_graph.pt"))
    torch.save(test_graph, os.path.join(DATA_DIR, "train_2_graph.pt"))
    with open(os.path.join(DATA_DIR, "ligand_dict.pkl"), "wb") as f:
        pickle.dump(ligand_dict, f)
    with open(os.path.join(DATA_DIR, "target_dict.pkl"), "wb") as f:
        pickle.dump(target_dict, f)

    logger.info("Graphs and dictionaries saved.")
    return train_graph, test_graph

def main():
    db_path = os.path.join(DATA_DIR, "chembl_35.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChEMBL database not found at {db_path}")
    process_chembl_35(db_path)

if __name__ == "__main__":
    main()
