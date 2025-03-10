# **DS5500-Project: Drug Discovery with Graph Neural Networks**
 **Predicting potential Parkinson’s treatments using GNNs and ZINC20 molecules**

## ** Overview**
This project focuses on:
- **Training a GNN model on ChEMBL 35 data**
- **Processing ZINC20 molecules**
- **Ranking potential drug candidates for Parkinson’s disease**

The pipeline automates downloading, preprocessing, training, and inference using **PyTorch Geometric**.

---

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
```
📂 DS5500-project/
├── 📂 src/                   # Source code
│   ├── 📂 preprocessing/      # Data processing scripts
│   │   ├── preprocess_data.py
│   │   ├── preprocess_PD_data.py
│   │   ├── process_zinc_data.py
│   ├── 📂 models/             # Model training & inference
│   │   ├── train_SAGEConv_model.py
│   │   ├── predict_zinc_data.py
│   ├── 📂 data_download/       # Scripts for downloading datasets
│   │   ├── download_zinc_data.py
│   │   ├── covert_zinc_data.py
│   ├── 📂 utils/               # Helper functions
│   │   ├── zinc20_process_predictions.py
├── 📂 data/                   # Raw & processed data
├── 📂 models/                 # Trained models
├── 📂 results/                # Final rankings & reports
├── 📜 README.md               # Project documentation
├── 📜 makefile                # Automates workflow
├── 📜 requirements.txt        # List of dependencies
```

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
