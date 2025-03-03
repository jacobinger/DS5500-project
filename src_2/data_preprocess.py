#!/usr/bin/env python3

# preprocess_bindingdb.py
# This script preprocesses the BindingDB dataset for a GNN project, focusing on ligand-target interactions
# for drug discovery, specifically targeting kinase proteins, with heterogeneous graph construction.

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
from Levenshtein import distance  # For simple sequence similarity (install with `pip install python-Levenshtein`)
from torch_geometric.data import Data

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for downloaded and extracted files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    """
    Download a file from a URL and save it to the specified filename.
    If the file already exists, check if it's valid (for ZIP files) and re-download if necessary.
    """
    if os.path.exists(filename):
        # For zip files, check validity
        if filename.endswith('.zip'):
            if not zipfile.is_zipfile(filename):
                logger.warning(f"{filename} exists but is not a valid zip file. Removing and re-downloading.")
                os.remove(filename)
            else:
                logger.info(f"{filename} already exists and is a valid zip file. Skipping download.")
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
    """
    Extract a ZIP file to the specified directory.
    If all files from the ZIP already exist in the target directory, skip extraction.
    """
    # Check if the zip file is valid before attempting extraction
    if not zipfile.is_zipfile(zip_path):
        logger.error(f"{zip_path} is not a valid zip file. Cannot extract.")
        raise zipfile.BadZipFile(f"{zip_path} is not a valid zip file.")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            # Check if every member already exists in the extraction directory.
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
    """
    Load and parse the BindingDB TSV file, filtering for kinase targets.
    Skips malformed lines and handles inconsistent column counts.
    Returns a DataFrame with standardized columns: 'LigandSMILES', 'TargetID', and 'IC50_nM'.
    """
    try:
        logger.info(f"Loading TSV file from {tsv_path}")
        # Load TSV with lenient parsing to skip bad lines, limit to 1,000 rows for testing
        df = pd.read_csv(tsv_path, sep='\t', engine='python', on_bad_lines='skip', dtype=str, nrows=1000)  # Adjust nrows as needed
        
        # Identify the target name column by looking for common patterns in column names
        target_name_cols = [col for col in df.columns if "targetname" in col.lower()]
        if not target_name_cols:
            target_name_cols = [col for col in df.columns if "target" in col.lower()]
        if not target_name_cols:
            logger.error("No target name column found in TSV file.")
            raise KeyError("No target name column found in TSV file.")
        target_name_col = target_name_cols[0]
        
        # Filter for kinase targets using the identified target name column
        kinase_targets = df[df[target_name_col].str.contains('kinase', case=False, na=False)]
        
        # Identify the ligand SMILES column by checking for both "ligand" and "smiles"
        ligand_smiles_cols = [col for col in df.columns if "ligand" in col.lower() and "smiles" in col.lower()]
        if not ligand_smiles_cols:
            logger.error("No ligand SMILES column found in TSV file.")
            raise KeyError("No ligand SMILES column found in TSV file.")
        ligand_smiles_col = ligand_smiles_cols[0]
        
        # Identify the target ID column:
        # First try for columns that contain both "primary id" and either "swissprot" or "trembl"
        target_id_cols = [col for col in df.columns if "primary id" in col.lower() and (("swissprot" in col.lower()) or ("trembl" in col.lower()))]
        if not target_id_cols:
            # Fallback: try using "monomerid" (ensure that this column corresponds to target IDs in your data)
            target_id_cols = [col for col in df.columns if "monomerid" in col.lower()]
        if not target_id_cols:
            logger.error("No target ID column found in TSV file.")
            raise KeyError("No target ID column found in TSV file.")
        target_id_col = target_id_cols[0]
        
        # Identify the affinity column (e.g., "IC50 (nM)" or similar)
        affinity_cols = [col for col in df.columns if "ic50" in col.lower() or "affinity" in col.lower()]
        if not affinity_cols:
            logger.error("No affinity column found in TSV file.")
            raise KeyError("No affinity column found in TSV file.")
        affinity_col = affinity_cols[0]
        
        # Select the desired columns and drop rows with missing critical values
        processed_df = kinase_targets[[ligand_smiles_col, target_id_col, affinity_col]].dropna(how='all')
        processed_df = processed_df.dropna(subset=[ligand_smiles_col, target_id_col, affinity_col])
        
        # Standardize column names for later processing
        processed_df = processed_df.rename(columns={
            ligand_smiles_col: "LigandSMILES",
            target_id_col: "TargetID",
            affinity_col: "IC50_nM"
        })
        
        logger.info(f"Loaded {len(processed_df)} kinase target measurements after skipping bad lines")
        return processed_df
    except Exception as e:
        logger.error(f"Error loading TSV file: {e}")
        raise

def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a PyTorch Geometric graph using RDKit.
    Returns a Data object with node features, edge indices, and edge features.
    """
    try:
        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles} - skipped")
            return None
        
        # Get atoms (nodes)
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),     # Number of bonds
                atom.GetHybridization().real if atom.GetHybridization() else 0,  # Hybridization
                atom.GetFormalCharge(),  # Charge
            ]
            atom_features.append(features)
        x = torch.tensor(atom_features, dtype=torch.float)  # Node features

        # Get bonds (edges)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected graph: add reverse edge
            bond_type = bond.GetBondTypeAsDouble()
            edge_attr.append([bond_type])
            edge_attr.append([bond_type])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Shape: [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Edge features

        # Create graph data object
        print('test')
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {e}")
        return None

def embed_target_sequence(sequence, max_length=1024):
    """
    Embed a protein sequence using ESM-2 (Evolutionary Scale Modeling 2) model.
    Returns a tensor of sequence features.
    """
    try:
        # Initialize ESM-2 model and tokenizer (pre-trained 650M parameter model)
        model_name = "facebook/esm2_t33_650M_UR50D"  # Large ESM-2 model for proteins
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)

        # Tokenize and embed the sequence
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            # Average the token embeddings to get a single vector per sequence
            embedding = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, hidden_size]
        
        return embedding.squeeze(0)  # Remove batch dimension
    except Exception as e:
        logger.error(f"Error embedding target sequence: {e}")
        return None

def load_fasta_data(fasta_path):
    """
    Load and parse the BindingDB target sequences from FASTA file.
    Returns a dictionary mapping target IDs to sequences.
    """
    try:
        logger.info(f"Loading FASTA file from {fasta_path}")
        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            # Assume the ID is in the format "TargetID|Sequence" or similar
            target_id = record.id.split('|')[0]  # Adjust based on actual FASTA format
            sequences[target_id] = str(record.seq)
        logger.info(f"Loaded {len(sequences)} target sequences")
        return sequences
    except Exception as e:
        logger.error(f"Error loading FASTA file: {e}")
        raise

def compute_sequence_similarity(seq1, seq2, threshold=0.7):
    """
    Compute sequence similarity using Levenshtein distance (normalized to 0-1).
    Returns True if similarity exceeds threshold, False otherwise.
    """
    try:
        if len(seq1) == 0 or len(seq2) == 0:
            return False
        # Normalize Levenshtein distance to get similarity (1 - distance/length)
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return False
        sim = 1 - (distance(seq1, seq2) / max_len)
        return sim >= threshold
    except Exception as e:
        logger.warning(f"Error computing sequence similarity: {e}")
        return False

def create_heterogeneous_graph(binding_data, target_sequences):
    """
    Create a heterogeneous graph with ligand and target nodes, binding edges, and target-target similarity edges.
    """
    try:
        logger.info("Creating heterogeneous graph")
        graph = HeteroData()

        # Track unique ligands and targets for indexing
        ligand_to_idx = {}
        target_to_idx = {}
        ligand_idx = 0
        target_idx = 0

        # Store binding edges and features
        binding_edges = []
        binding_edge_attrs = []

        # Process each binding measurement
        for _, row in binding_data.iterrows():
            smiles = row['LigandSMILES']
            target_id = row['TargetID']
            affinity = row['IC50_nM']  # Adjust column name as needed

            # Create or get ligand graph
            if smiles not in ligand_to_idx:
                ligand_graph = smiles_to_graph(smiles)
                if ligand_graph is None:
                    continue
                graph['ligand'][ligand_idx] = ligand_graph
                ligand_to_idx[smiles] = ligand_idx
                ligand_idx += 1
            ligand_idx_current = ligand_to_idx[smiles]

            # Create or get target embedding
            if target_id in target_sequences:
                target_seq = target_sequences[target_id]
                target_embedding = embed_target_sequence(target_seq)
                if target_embedding is None:
                    continue
                if target_id not in target_to_idx:
                    graph['target'][target_idx] = Data(x=target_embedding.unsqueeze(0))  # Ensure shape [1, hidden_size]
                    target_to_idx[target_id] = target_idx
                    target_idx += 1
                target_idx_current = target_to_idx[target_id]

                # Add binding edge (ligand -> target) with affinity as edge feature
                binding_edges.append([ligand_idx_current, target_idx_current])
                binding_edge_attrs.append([float(affinity) if pd.notna(affinity) else 0.0])  # Default to 0 if NaN

        # Add binding edges to graph
        if binding_edges:
            graph['ligand', 'binds_to', 'target'].edge_index = torch.tensor(binding_edges, dtype=torch.long).t()
            graph['ligand', 'binds_to', 'target'].edge_attr = torch.tensor(binding_edge_attrs, dtype=torch.float)

        # Add target-target edges based on sequence similarity
        target_ids = list(target_to_idx.keys())
        target_edges = []
        target_edge_attrs = []  # Optional: store similarity score as edge feature

        for i, target_id1 in enumerate(target_ids):
            seq1 = target_sequences[target_id1]
            idx1 = target_to_idx[target_id1]
            for j, target_id2 in enumerate(target_ids[i+1:], start=i+1):
                seq2 = target_sequences[target_id2]
                idx2 = target_to_idx[target_id2]
                if compute_sequence_similarity(seq1, seq2):
                    target_edges.append([idx1, idx2])
                    target_edges.append([idx2, idx1])  # Undirected edge
                    target_edge_attrs.append([1.0])  # Similarity score (1 for similar, 0 otherwise)
                    target_edge_attrs.append([1.0])

        if target_edges:
            graph['target', 'similar_to', 'target'].edge_index = torch.tensor(target_edges, dtype=torch.long).t()
            graph['target', 'similar_to', 'target'].edge_attr = torch.tensor(target_edge_attrs, dtype=torch.float)

        logger.info(f"Created heterogeneous graph with {len(ligand_to_idx)} ligands and {len(target_to_idx)} targets")
        return graph
    except Exception as e:
        logger.error(f"Error creating heterogeneous graph: {e}")
        raise

def main():
    """
    Main function to preprocess BindingDB data for GNN with heterogeneous graph construction.
    """
    # URLs for BindingDB files (as of February 2025)
    TSV_URL = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All_202503_tsv.zip"
    FASTA_URL = "https://bindingdb.org/rwd/bind/BindingDBTargetSequences.fasta"

    # Download and extract files
    tsv_zip_path = os.path.join(DATA_DIR, "BindingDB_All_202502_tsv.zip")
    fasta_path = os.path.join(DATA_DIR, "BindingDBTargetSequences.fasta")

    # Download TSV zip file if needed, then extract if valid and not already extracted.
    download_file(TSV_URL, tsv_zip_path)
    extract_zip(tsv_zip_path, DATA_DIR)
    tsv_file = os.path.join(DATA_DIR, "BindingDB_All.tsv")  # Adjust based on the extracted filename

    # Download FASTA file if needed.
    download_file(FASTA_URL, fasta_path)

    # Load and process TSV data
    binding_data = load_tsv_data(tsv_file)

    # Load target sequences
    target_sequences = load_fasta_data(fasta_path)

    # Create heterogeneous graph
    graph = create_heterogeneous_graph(binding_data, target_sequences)

    # Save the heterogeneous graph
    torch.save(graph, os.path.join(DATA_DIR, "bindingdb_hetero_graph.pt"))
    logger.info("Saved heterogeneous graph to data/bindingdb_hetero_graph.pt")

if __name__ == "__main__":
    main()