import os
import sqlite3
import logging
import torch
import numpy as np
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load pretrained models for target embeddings
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model.eval()

def get_ligand_features(smiles):
    """Generate ECFP4 fingerprints for ligands."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = fp_gen.GetFingerprint(mol)
    logger.info(f"Generated ECFP for {smiles[:20]}...: shape {np.array(fp).shape}")
    return np.array(fp, dtype=np.float32)

def get_target_embedding(sequence):
    """Generate ESM-2 embeddings for target protein sequences."""
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    # Average over the sequence length dimension to obtain a fixed-size embedding.
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def process_chembl_35(db_path, output_path, max_pairs=200000):
    """Process ChEMBL 35 database into a heterogeneous graph, filtered for PD targets."""
    # Connect to the ChEMBL database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for required tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {t[0] for t in cursor.fetchall()}
    required_tables = {
        'molecule_dictionary', 'compound_structures', 'activities',
        'assays', 'target_dictionary', 'target_components', 'component_sequences'
    }
    if not required_tables.issubset(tables):
        missing = required_tables - tables
        raise ValueError(f"Missing required tables: {missing}")
    
    # SQL query to extract PD-specific ligand-target pairs
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
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences csq ON tc.component_id = csq.component_id
    WHERE act.standard_value IS NOT NULL
        AND act.standard_type IN ('IC50', 'Kd', 'Ki')
        AND act.standard_units = 'nM'
        AND cs.canonical_smiles IS NOT NULL
        AND csq.sequence IS NOT NULL
        AND td.chembl_id IN ('CHEMBL1795186', 'CHEMBL6151', 'CHEMBL2176839', 'CHEMBL5408', 'CHEMBL6122')
    ORDER BY act.activity_id
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(max_pairs,))
    conn.close()

    logger.info(f"Extracted {len(df)} PD-specific ligand-target pairs")
    if len(df) == 0:
        logger.warning("No data extracted. Check PD target filters or database content.")
    else:
        logger.info(f"Sample row: {df.iloc[0].to_dict()}")

    # Initialize the heterogeneous graph and dictionaries to track nodes
    graph = HeteroData()
    ligand_dict = {}
    target_dict = {}
    edge_list = []
    edge_attrs = []

    # Process each row from the DataFrame
    for idx, row in df.iterrows():
        ligand_id = row["ligand_id"]
        target_id = row["target_id"]
        
        # Compute ligand features if not already processed
        if ligand_id not in ligand_dict:
            feats = get_ligand_features(row["smiles"])
            if feats is not None:
                ligand_dict[ligand_id] = (len(ligand_dict), feats)
        
        # Compute target embeddings if not already processed
        if target_id not in target_dict:
            embedding = get_target_embedding(row["target_sequence"])
            target_dict[target_id] = (len(target_dict), embedding)
        
        # Add an edge if both ligand and target features are available
        if ligand_id in ligand_dict and target_id in target_dict:
            ligand_idx = ligand_dict[ligand_id][0]
            target_idx = target_dict[target_id][0]
            # Convert affinity (nM) to a negative log scale (pIC50/pKd/pKi)
            affinity = -np.log10(row["affinity"] / 1e9 + 1e-10)
            edge_list.append([ligand_idx, target_idx])
            edge_attrs.append(affinity)

    # Populate graph with node features and edge data
    graph["ligand"].x = torch.tensor([val[1] for val in ligand_dict.values()], dtype=torch.float)
    graph["target"].x = torch.tensor([val[1] for val in target_dict.values()], dtype=torch.float)
    graph["ligand", "binds_to", "target"].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    graph["ligand", "binds_to", "target"].edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(-1)

    # Log graph details
    logger.info(f"Ligand features shape: {graph['ligand'].x.shape}")
    logger.info(f"Target features shape: {graph['target'].x.shape}")
    logger.info(f"Edge index shape: {graph['ligand', 'binds_to', 'target'].edge_index.shape}")
    logger.info(f"Edge attr stats: min={graph['ligand', 'binds_to', 'target'].edge_attr.min().item()}, "
                f"max={graph['ligand', 'binds_to', 'target'].edge_attr.max().item()}, "
                f"mean={graph['ligand', 'binds_to', 'target'].edge_attr.mean().item()}")

    # Save the heterogeneous graph to disk
    torch.save(graph, output_path)
    logger.info(f"Saved PD-specific graph to {output_path}")
    return graph

def main():
    """Main function to process ChEMBL 35 data and generate a heterogeneous graph."""
    db_path = os.path.join(DATA_DIR, "chembl_35.db")
    output_path = os.path.join(DATA_DIR, "chembl_35_hetero_graph_PD_3.pt")
    
    if os.path.exists(output_path):
        logger.info(f"Graph already exists at {output_path}. Skipping processing.")
    else:
        process_chembl_35(db_path, output_path, max_pairs=2000000000)

if __name__ == "__main__":
    main()
