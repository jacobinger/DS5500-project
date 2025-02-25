from rdkit import Chem
import gzip
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sdf_gz(sdf_path, sanitize=True, remove_hs=True):
    """
    Load and parse a compressed SDF (.sdf.gz) file using RDKit.
    Returns a list of tuples (RDKit molecule, ChEMBL ID) for valid molecules.
    """
    molecules = []
    try:
        with gzip.open(sdf_path, 'rb') as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=sanitize, removeHs=remove_hs)
            for i, mol in enumerate(supplier):
                if mol is None:
                    logger.warning(f"Invalid molecule at line {i + 1} - skipped due to sanitization error")
                    continue
                
                # Try to get ChEMBL ID, handle missing or empty values
                try:
                    chembl_id = mol.GetProp('chembl_id').strip()
                    if not chembl_id:
                        logger.warning(f"No ChEMBL ID for molecule at line {i + 1} - skipped")
                        continue
                except KeyError:
                    logger.warning(f"ChEMBL ID missing for molecule at line {i + 1} - skipped")
                    continue
                
                molecules.append((mol, chembl_id))
        logger.info(f"Loaded {len(molecules)} valid molecules from {sdf_path}")
    except Exception as e:
        logger.error(f"Error loading SDF file: {e}")
    return molecules

# Specify the path to your SDF file
sdf_path = 'data/chembl_35.sdf.gz'
molecules_with_ids = load_sdf_gz(sdf_path, sanitize=True, remove_hs=True)

from chembl_webresource_client.new_client import new_client
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Initialize ChEMBL client
target = new_client.target
activity = new_client.activity
molecule = new_client.molecule

def fetch_target_info(chembl_id, max_retries=3, delay=2):
    """
    Fetch target and activity data for a given ChEMBL ID from the ChEMBL database,
    with retry logic for rate limits or connectivity issues.
    """
    for attempt in range(max_retries):
        try:
            # Get molecule data (optional, for validation)
            mol_data = molecule.get(chembl_id)
            
            # Get activities (e.g., IC50, Ki) and associated targets
            activities = activity.filter(molecule_chembl_id=chembl_id).only(['standard_value', 'standard_units', 'target_chembl_id', 'assay_chembl_id', 'target_pref_name'])
            
            # Get target details
            target_info = []
            for act in activities:
                target_id = act['target_chembl_id']
                if target_id:
                    try:
                        target_data = target.get(target_id)
                        target_info.append({
                            'target_chembl_id': target_id,
                            'target_name': target_data['pref_name'],
                            'target_type': target_data['target_type'],
                            'organism': target_data['organism'],
                            'accession': target_data['accession'],
                            'activity_value': act['standard_value'],
                            'activity_units': act['standard_units']
                        })
                    except Exception as e:
                        logger.warning(f"Error fetching target data for target_chembl_id {target_id}: {e}")
                        continue
            
            if target_info:
                logger.info(f"Found {len(target_info)} targets for ChEMBL ID {chembl_id}")
            else:
                logger.info(f"No target/activity data found for ChEMBL ID {chembl_id}")
            return target_info
        
        except Exception as e:
            if 'Rate limit' in str(e) or 'ConnectionError' in str(e):
                logger.warning(f"Rate limit or connection issue for ChEMBL ID {chembl_id} (attempt {attempt + 1}/{max_retries}) - retrying in {delay} seconds")
                time.sleep(delay)
                continue
            logger.error(f"Error fetching data for ChEMBL ID {chembl_id}: {e}")
            return []

# Example usage with your molecules, including logging and printing
for mol, chembl_id in molecules_with_ids[:5]:  # Test with the first 5 molecules
    logger.info(f"Processing ChEMBL ID: {chembl_id}")
    target_data = fetch_target_info(chembl_id)
    print(f"ChEMBL ID: {chembl_id}")
    for t in target_data:
        print(f"  Target: {t['target_name']} (ID: {t['target_chembl_id']}, Type: {t['target_type']}, Activity: {t['activity_value']} {t['activity_units']})")
