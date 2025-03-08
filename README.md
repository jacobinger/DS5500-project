# DS5500-project

## Getting Started

### Prerequisites
- Ensure you have `wget` and Python 3 installed.
- Install dependencies for `zinc_data.py`:
  ```bash
  conda create -n conda_usml_env python=3.9
  conda activate conda_usml_env
  pip install rdkit numpy



## TODO/net steps
1. use all of the zinc data downloaded,
2. find more data from the chembl35 to train on.

3. Analyze Top Candidates

Cross-check these compounds in PubChem or ChEMBL.
Look for existing research on similar molecules.
4. Perform Molecular Docking Simulations

Use AutoDock Vina or SwissDock to simulate binding to Parkinson’s-related targets.
Check binding affinity and interactions.
5. Apply Further Filtering

Use ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) models.
Remove toxic or non-drug-like compounds.
6. Compare with Known Parkinson’s Drugs

Find similar molecules to existing FDA-approved drugs.
