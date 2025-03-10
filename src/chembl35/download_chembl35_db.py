import os
import logging
import requests
import tarfile

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global settings
DATA_DIR = "data"
CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.sdf.gz"
CHEMBL_TAR = os.path.join(DATA_DIR, "chembl_35.sdf.gz")
OUTPUT_DB = os.path.join(DATA_DIR, "chembl_35.db")

def download_chembl_tar(url, output_path):
    """Download the ChEMBL 35 SQLite tarball if not already present."""
    if os.path.exists(output_path):
        logger.info(f"ChEMBL tarball already exists at {output_path}, skipping download")
        return True
    logger.info(f"Downloading ChEMBL 35 data from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded ChEMBL tarball to {output_path}")
        return True
    else:
        logger.error(f"Failed to download ChEMBL data: {response.status_code}")
        return False

def extract_chembl_tar(tar_path, extract_dir):
    """Extract the tarball to produce chembl_35.db if not already extracted."""
    if os.path.exists(OUTPUT_DB):
        logger.info(f"ChEMBL database already exists at {OUTPUT_DB}, skipping extraction")
        return True
    if not os.path.exists(tar_path):
        logger.error(f"Tarball not found at {tar_path}")
        return False
    logger.info(f"Extracting {tar_path} to {extract_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    logger.info(f"Extracted ChEMBL data to {extract_dir}")
    return True

def cleanup(tar_path):
    """Remove the tarball to save space (optional)."""
    if os.path.exists(tar_path):
        os.remove(tar_path)
        logger.info(f"Removed {tar_path}")
    else:
        logger.info(f"No tarball found at {tar_path} to clean up")

def main():
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Download the tarball if not present
    if not download_chembl_tar(CHEMBL_URL, CHEMBL_TAR):
        raise RuntimeError("Failed to download ChEMBL tarball")

    # Step 2: Extract it if not already extracted
    if not extract_chembl_tar(CHEMBL_TAR, DATA_DIR):
        raise RuntimeError("Failed to extract ChEMBL tarball")

    # Step 3: Optional cleanup
    cleanup(CHEMBL_TAR)

if __name__ == "__main__":
    main()