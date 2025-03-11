# Define paths
SRC_DIR = src
DATA_DIR = data
MODEL_DIR = models
RESULTS_DIR = results

# Define scripts
DOWNLOAD_SCRIPT = $(SRC_DIR)/download_zinc_data.py
CONVERT_SCRIPT = $(SRC_DIR)/covert_zinc_data.py
PREPROCESS_SCRIPT = $(SRC_DIR)/preprocess_data.py
PD_PREPROCESS_SCRIPT = $(SRC_DIR)/preprocess_PD_data.py
PROCESS_ZINC_SCRIPT = $(SRC_DIR)/process_zinc_data.py
TRAIN_SCRIPT = $(SRC_DIR)/train_SAGEConv_model.py
PREDICT_SCRIPT = $(SRC_DIR)/predict_zinc_data.py
RANK_SCRIPT = $(SRC_DIR)/zinc20_process_predictions.py

# Define output files
ZINC_DATA = $(DATA_DIR)/zinc20_hetero_graph.pt
MODEL_FILE = $(MODEL_DIR)/gnn_model.pt
PREDICTIONS = $(RESULTS_DIR)/zinc20_predictions.csv

# Python executable
PYTHON = python

# Default target
all: download convert preprocess process train predict rank

# Step 1: Download ZINC20 Data
download:
	$(PYTHON) $(DOWNLOAD_SCRIPT)

# Step 2: Convert ZINC Data (if needed)
convert:
	$(PYTHON) $(CONVERT_SCRIPT)

# Step 3: Preprocess General Data
preprocess:
	$(PYTHON) $(PREPROCESS_SCRIPT)

# Step 4: Preprocess Parkinson's Data
preprocess_pd:
	$(PYTHON) $(PD_PREPROCESS_SCRIPT)

# Step 5: Process ZINC Data
process:
	$(PYTHON) $(PROCESS_ZINC_SCRIPT)

# Step 6: Train Model
train:
	$(PYTHON) $(TRAIN_SCRIPT)

# Step 7: Run Predictions on ZINC20
predict:
	$(PYTHON) $(PREDICT_SCRIPT)

# Step 8: Rank & Filter Predictions
rank:
	$(PYTHON) $(RANK_SCRIPT)

# Clean intermediate files
clean:
	rm -f $(ZINC_DATA) $(MODEL_FILE) $(PREDICTIONS)

preprocess_retrosynth:
	python src/retrosynthesis/preprocess_for_retrosynth.py

run_retrosynth:
	python src/retrosynthesis/retrosynth_analyze.py

train_gcn:
	python src/retrosynthesis/train_gcn_model.py

gcn_predict:
	python src/retrosynthesis/gcn_retrosynth_predict.py

validate_routes:
	python src/retrosynthesis/validate_synth_routes.py

# Run the full pipeline
run: all
	echo "Pipeline Completed Successfully!"

