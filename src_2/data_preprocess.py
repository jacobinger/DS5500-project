#!/usr/bin/env python3
import os
import zipfile
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import HeteroData  # For heterogeneous graphs
import logging
from Bio import SeqIO  # For parsing FASTA files
from transformers import AutoTokenizer, EsmModel
import numpy as np
from Levenshtein import distance  # For simple sequence similarity (pip install python-Levenshtein)
from torch_geometric.data import Data
from requests.exceptions import ReadTimeout
import re

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for downloaded and extracted files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    if os.path.exists(filename):
        if filename.endswith('.zip'):
            if not zipfile.is_zipfile(filename):
                logger.warning(f"{filename} exists but is not a valid zip file. Removing and re-downloading.")
                os.remove(filename)
            else:
                logger.info(f"{filename} already exists and is valid. Skipping download.")
                return
        else:
            logger.info(f"{filename} already exists. Skipping download.")
            return
    try:
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded {filename} successfully")
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        raise

def extract_zip(zip_path, extract_dir):
    if not zipfile.is_zipfile(zip_path):
        logger.error(f"{zip_path} is not a valid zip file. Cannot extract.")
        raise zipfile.BadZipFile(f"{zip_path} is not a valid zip file.")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            if all(os.path.exists(os.path.join(extract_dir, member)) for member in members):
                logger.info(f"{zip_path} already extracted to {extract_dir}. Skipping extraction.")
                return
            logger.info(f"Extracting {zip_path} to {extract_dir}")
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted {zip_path} successfully")
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        raise

def load_tsv_data(tsv_path):
    try:
        logger.info(f"Loading TSV file from {tsv_path}")
        # Load a subset (e.g. first 5000 lines) for speed.
        df = pd.read_csv(tsv_path, sep='\t', engine='python', on_bad_lines='skip', dtype=str, nrows=5000)
        logger.info(f"TSV columns: {df.columns.tolist()}")
        # Identify a target name column (e.g., one that contains 'targetname' or 'target')
        target_name_cols = [col for col in df.columns if "targetname" in col.lower()]
        if not target_name_cols:
            target_name_cols = [col for col in df.columns if "target" in col.lower()]
        if not target_name_cols:
            logger.error("No target name column found in TSV file.")
            raise KeyError("No target name column found in TSV file.")
        target_name_col = target_name_cols[0]
        logger.info(f"Sample target names: {df[target_name_col].head().tolist()}")
        # Filter for kinase targets using a regex on the target name
        kinase_targets = df[df[target_name_col].str.contains('kinase|protein kinase|tyrosine kinase|serine/threonine kinase|pkc|mapk|cdk|akt', case=False, na=False)]
        logger.info(f"Rows before filtering: {len(df)}, after filtering: {len(kinase_targets)}")
        # Identify ligand SMILES, target ID, and affinity columns.
        ligand_smiles_cols = [col for col in df.columns if "ligand" in col.lower() and "smiles" in col.lower()]
        if not ligand_smiles_cols:
            logger.error("No ligand SMILES column found in TSV file.")
            raise KeyError("No ligand SMILES column found in TSV file.")
        ligand_smiles_col = ligand_smiles_cols[0]
        target_id_cols = [col for col in df.columns if "primary id" in col.lower() and (("swissprot" in col.lower()) or ("trembl" in col.lower()))]
        if not target_id_cols:
            target_id_cols = [col for col in df.columns if "monomerid" in col.lower() or "target id" in col.lower()]
        if not target_id_cols:
            logger.error("No target ID column found in TSV file.")
            raise KeyError("No target ID column found in TSV file.")
        target_id_col = target_id_cols[0]
        affinity_cols = [col for col in df.columns if "ic50" in col.lower() or "affinity" in col.lower()]
        if not affinity_cols:
            logger.error("No affinity column found in TSV file.")
            raise KeyError("No affinity column found in TSV file.")
        affinity_col = affinity_cols[0]
        # Select desired columns and drop rows with missing values.
        processed_df = kinase_targets[[ligand_smiles_col, target_id_col, affinity_col, target_name_col]].dropna(how='all')
        processed_df = processed_df.dropna(subset=[ligand_smiles_col, target_id_col, affinity_col])
        logger.info(f"Final DataFrame sample:\n{processed_df.head()}")
        logger.info(f"Sample TargetIDs: {processed_df[target_id_col].head().tolist()}")
        # Standardize column names.
        processed_df = processed_df.rename(columns={
            ligand_smiles_col: "LigandSMILES",
            target_id_col: "TargetID",
            affinity_col: "IC50_nM",
            target_name_col: "TargetName"
        })
        logger.info(f"Loaded {len(processed_df)} kinase target measurements after filtering")
        return processed_df
    except Exception as e:
        logger.error(f"Error loading TSV file: {e}")
        raise

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles} - skipped")
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetHybridization().real if atom.GetHybridization() else 0,
                atom.GetFormalCharge()
            ]
            atom_features.append(features)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attr.append([bond_type])
            edge_attr.append([bond_type])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {e}")
        return None

def embed_target_sequences_batch(sequences, max_length=1024):
    """
    Batch-process a list of protein sequences using ESM2.
    Returns a tensor of embeddings, one per sequence.
    """
    try:
        # Use MPS if available (for Apple Silicon)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name).to(device)
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.to("cpu")
    except Exception as e:
        logger.error(f"Error embedding target sequences in batch: {e}")
        return None

def embed_target_sequence(sequence, max_length=1024):
    # Fallback to single-sequence embedding if needed.
    try:
        return embed_target_sequences_batch([sequence], max_length=max_length)[0]
    except Exception as e:
        logger.error(f"Error embedding target sequence: {e}")
        return None

def parse_fasta_header(header):
    parts = header.split()
    for i, part in enumerate(parts):
        if part.startswith("length:"):
            protein_name = " ".join(parts[i+1:])
            return protein_name.strip()
    return ""

def load_fasta_data_with_names(fasta_path):
    sequences = {}
    names = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        fasta_id = record.id
        sequences[fasta_id] = str(record.seq)
        protein_name = parse_fasta_header(record.description)
        names[fasta_id] = protein_name
    return sequences, names

def compute_sequence_similarity(seq1, seq2, threshold=0.7):
    try:
        if len(seq1) == 0 or len(seq2) == 0:
            return False
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return False
        sim = 1 - (distance(seq1, seq2) / max_len)
        return sim >= threshold
    except Exception as e:
        logger.warning(f"Error computing sequence similarity: {e}")
        return False

def fetch_target_mappings(url="https://www.bindingdb.org/rwd/bind/ByUniProtids.jsp", timeout=60):
    try:
        logger.info(f"Fetching target mappings from {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        mappings = {}  # Add parsing logic if needed.
        logger.info(f"Loaded {len(mappings)} target mappings from web")
        return mappings
    except ReadTimeout as e:
        logger.error(f"Timeout error fetching target mappings: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching target mappings from {url}: {e}")
        return {}

def create_target_mapping(tsv_target_names, fasta_names):
    mapping = {}
    for t_name in tsv_target_names.unique():
        t_name_lower = t_name.lower()
        best_match = None
        best_score = 0
        for fasta_id, fasta_name in fasta_names.items():
            fasta_name_lower = fasta_name.lower()
            words = t_name_lower.split()
            if len(words) == 0:
                continue
            score = sum(1 for w in words if w in fasta_name_lower) / len(words)
            if score > best_score:
                best_score = score
                best_match = fasta_id
        mapping[t_name] = best_match if best_score > 0.3 else None
    return mapping

def parse_affinity(affinity_str):
    try:
        m = re.search(r"(\d+\.?\d*)", affinity_str)
        if m:
            return float(m.group(1))
    except Exception as e:
        logger.warning(f"Error parsing affinity '{affinity_str}': {e}")
    return 0.0

def create_heterogeneous_graph(binding_data, target_sequences, target_mappings=None):
    try:
        logger.info("Creating heterogeneous graph")
        graph = HeteroData()
        ligand_to_idx = {}
        target_to_idx = {}
        ligand_idx = 0
        target_idx = 0
        binding_edges = []
        binding_edge_attrs = []

        # Precompute batch embeddings for all unique targets needed:
        unique_fasta_ids = set()
        for _, row in binding_data.iterrows():
            tsv_target_name = row['TargetName']
            if target_mappings:
                fasta_id = target_mappings.get(tsv_target_name)
                if fasta_id and fasta_id in target_sequences:
                    unique_fasta_ids.add(fasta_id)
        unique_fasta_ids = sorted(list(unique_fasta_ids))
        logger.info(f"Unique target FASTA IDs to embed: {unique_fasta_ids}")
        embedding_dict = {}
        if unique_fasta_ids:
            unique_sequences = [target_sequences[fid] for fid in unique_fasta_ids]
            batch_embeddings = embed_target_sequences_batch(unique_sequences, max_length=1024)
            if batch_embeddings is None:
                logger.error("Failed to compute batch embeddings")
            else:
                embedding_dict = {fid: emb for fid, emb in zip(unique_fasta_ids, batch_embeddings)}

        # Now construct graph by iterating over binding data:
        for _, row in binding_data.iterrows():
            smiles = row['LigandSMILES']
            tsv_target_name = row['TargetName']
            affinity = row['IC50_nM']
            # Process ligand
            if smiles not in ligand_to_idx:
                ligand_graph = smiles_to_graph(smiles)
                if ligand_graph is None:
                    continue
                if 'x' not in graph['ligand']:
                    graph['ligand'].x = ligand_graph.x
                    graph['ligand'].edge_index = ligand_graph.edge_index
                    graph['ligand'].edge_attr = ligand_graph.edge_attr
                else:
                    graph['ligand'].x = torch.cat([graph['ligand'].x, ligand_graph.x], dim=0)
                    graph['ligand'].edge_index = torch.cat([graph['ligand'].edge_index, ligand_graph.edge_index], dim=1)
                    graph['ligand'].edge_attr = torch.cat([graph['ligand'].edge_attr, ligand_graph.edge_attr], dim=0)
                ligand_to_idx[smiles] = ligand_idx
                ligand_idx += 1
            ligand_idx_current = ligand_to_idx[smiles]
            # Map TSV target name to FASTA ID
            fasta_id = None
            if target_mappings:
                fasta_id = target_mappings.get(tsv_target_name)
            if fasta_id is None:
                logger.warning(f"No mapping found for target name '{tsv_target_name}' - skipping")
                continue
            if fasta_id not in embedding_dict:
                logger.warning(f"No embedding found for FASTA id {fasta_id} (mapped from '{tsv_target_name}') - skipping")
                continue
            target_embedding = embedding_dict[fasta_id].unsqueeze(0)
            if tsv_target_name not in target_to_idx:
                if 'x' not in graph['target']:
                    graph['target'].x = target_embedding
                else:
                    graph['target'].x = torch.cat([graph['target'].x, target_embedding], dim=0)
                target_to_idx[tsv_target_name] = target_idx
                target_idx += 1
            target_idx_current = target_to_idx[tsv_target_name]
            binding_edges.append([ligand_idx_current, target_idx_current])
            binding_edge_attrs.append([parse_affinity(affinity) if pd.notna(affinity) else 0.0])
        
        if binding_edges:
            graph['ligand', 'binds_to', 'target'].edge_index = torch.tensor(binding_edges, dtype=torch.long).t().contiguous()
            graph['ligand', 'binds_to', 'target'].edge_attr = torch.tensor(binding_edge_attrs, dtype=torch.float)
        
        # (Optional) Add target-target similarity edges if desired.
        logger.info(f"Created heterogeneous graph with {len(ligand_to_idx)} ligands and {len(target_to_idx)} targets")
        return graph
    except Exception as e:
        logger.error(f"Error creating heterogeneous graph: {e}")
        raise

def main():
    TSV_URL = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All_202503_tsv.zip"
    FASTA_URL = "https://www.bindingdb.org/rwd/bind/BindingDBTargetSequences.fasta"
    UNIPROT_URL = "https://www.bindingdb.org/rwd/bind/ByUniProtids.jsp"  # Not used in this example
    
    tsv_zip_path = os.path.join(DATA_DIR, "BindingDB_All_202502_tsv.zip")
    fasta_path = os.path.join(DATA_DIR, "BindingDBTargetSequences.fasta")
    
    download_file(TSV_URL, tsv_zip_path)
    extract_zip(tsv_zip_path, DATA_DIR)
    tsv_file = os.path.join(DATA_DIR, "BindingDB_All.tsv")
    
    download_file(FASTA_URL, fasta_path)
    
    binding_data = load_tsv_data(tsv_file)
    logger.info(f"Binding data sample:\n{binding_data.head()}")
    
    target_sequences, fasta_names = load_fasta_data_with_names(fasta_path)
    logger.info(f"Some FASTA keys: {list(target_sequences.keys())[:100]}")
    
    tsv_target_names = binding_data["TargetName"]
    name_mapping = create_target_mapping(tsv_target_names, fasta_names)
    logger.info(f"Mapping from TSV target names to FASTA IDs:\n{name_mapping}")
    
    graph = create_heterogeneous_graph(binding_data, target_sequences, target_mappings=name_mapping)
    
    output_path = os.path.join(DATA_DIR, "bindingdb_hetero_graph.pt")
    torch.save(graph, output_path)
    logger.info(f"Saved heterogeneous graph to {output_path}")

if __name__ == "__main__":
    main()
