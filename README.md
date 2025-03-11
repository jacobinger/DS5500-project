# **DS5500-Project: Drug Discovery with Graph Neural Networks**
 **Predicting potential Parkinson’s treatments using GNNs and ZINC20 molecules**

## ** Overview**
This project focuses on:
- **Training a GNN model on ChEMBL 35 data**
- **Processing ZINC20 molecules**
- **Ranking potential drug candidates for Parkinson’s disease**

The pipeline automates downloading, preprocessing, training, and inference using **PyTorch Geometric**.

### Workflow
1. **Download Data**: Already done (`chembl35.db` and ZINC20 raw files).
2. **Preprocess ChEMBL35**: `make preprocess_pd` → `data/chembl35/pd_targets.csv`.
3. **Process ZINC20**: `make process_zinc` → `data/zinc20/zinc20_processed.csv`.
4. **Prepare Targets**: `make preprocess_retrosynth` → `data/retrosynth/target_molecules.csv`.
5. **Run Retrosynthesis**: `make run_retrosynth` → `data/retrosynth/synthetic_routes/routes.csv`.
6. **Validate Routes**: `make validate_routes` → `results/retrosynth_results/ranked_routes.csv`.

---
## Retrosynthetic Analysis with GCN
Uses ChEMBL35 for targets, ZINC20 for precursors, and a GCN to score routes.

### Notes
- **Retrosynthesis Tool**:

## ** Getting Started**

### ** Prerequisites**
- **Python 3.11** (recommended)
- **Conda** for environment management
- **Required packages** (`rdkit`, `numpy`, `torch_geometric`, etc.)

### ** Setup**
#### **1️ Create & Activate Virtual Environment**
```bash
conda create -n conda_usml_env python=3.13
conda activate conda_usml_env
```

#### **2️ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## ** Project Structure**

---

## ** Usage**
### **1️ Download ZINC20 Data**
```bash
make download
```

### **2️ Preprocess Data**
```bash
make preprocess
make preprocess_pd  # Parkinson’s specific processing
```

### **3️ Process & Structure ZINC20**
```bash
make process
```

### **4️ Train the GNN Model**
```bash
make train
```

### **5️ Run Predictions on ZINC20**
```bash
make predict
```

### **6️ Rank & Filter Drug Candidates**
```bash
make rank
```

## Retrosynthetic Analysis
This module enables retrosynthetic planning for novel drug candidates identified from ZINC20 predictions.

### Prerequisites
- Install additional dependencies: `pip install -r requirements.txt`
- (Optional) Set up ASKCOS or another retrosynthesis tool.

### Usage
1. Run ZINC20 prediction pipeline: `make predict_zinc`
2. Run retrosynthesis pipeline: `make retrosynth_pipeline`
3. Results are saved in `results/retrosynth_results/`.


### **7️ Run the Full Pipeline**
```bash
make run
```

---

## ** Results**
- Outputs a **ranked list of potential Parkinson’s drug candidates**.
- Results are saved in:  
   **`results/zinc20_predictions.csv`**
  
---

## ** Next Steps**
 **Perform Molecular Docking**  
 **Filter with ADMET properties**  
 **Compare with existing Parkinson’s drugs**

---

## ** Contributors**
- **Jake Inger** ([@jacobinger](https://github.com/jacobinger))

---

## ** License**
This project is open-source and available under the **MIT License**.

---

 **Now you have a professional README!** Let me know if you want to add more details. 


## TODO/net steps
!! REORG FILES TO MATCH STRUCTURE FILE !!

1. use all of the zinc data downloaded,
2. find more data from the chembl35 to train on.
- go through current prediction and see if they have any relevance 
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
