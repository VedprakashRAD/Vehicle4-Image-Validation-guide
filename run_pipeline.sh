#!/bin/bash

# Vehicle Orientation Detection Pipeline
# This script runs the full pipeline from data collection to model training

# Set up error handling
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="pipeline_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    log "ERROR: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Install required packages
log "Installing required packages..."
pip3 install -q tensorflow numpy opencv-python matplotlib scikit-learn tqdm simple_image_download pillow seaborn

# Step 1: Collect dataset
log "Step 1: Collecting dataset..."
python3 collect_large_dataset.py
if [ $? -ne 0 ]; then
    log "ERROR: Dataset collection failed."
    exit 1
fi

# Step 2: Preprocess dataset
log "Step 2: Preprocessing dataset..."
python3 preprocess_large_dataset.py
if [ $? -ne 0 ]; then
    log "ERROR: Dataset preprocessing failed."
    exit 1
fi

# Step 3: Train model
log "Step 3: Training model..."
python3 train_improved_model.py
if [ $? -ne 0 ]; then
    log "ERROR: Model training failed."
    exit 1
fi

# Step 4: Download test images
log "Step 4: Downloading test images..."
python3 download_test_images.py
if [ $? -ne 0 ]; then
    log "WARNING: Test image download failed. Continuing with pipeline."
fi

# Step 5: Test model
log "Step 5: Testing model..."
# Find the most recent model directory
MODEL_DIR=$(find . -type d -name "orientation_model_*" | sort -r | head -n 1)
if [ -z "$MODEL_DIR" ]; then
    log "ERROR: No model directory found."
    exit 1
fi

# Find the best model file
MODEL_FILE="${MODEL_DIR}/best_model.h5"
if [ ! -f "$MODEL_FILE" ]; then
    log "ERROR: Model file not found in ${MODEL_DIR}."
    exit 1
fi

# Run the test script
python3 test_orientation_model.py --model "$MODEL_FILE" --test_dir "test_images" --output_dir "test_results"
if [ $? -ne 0 ]; then
    log "ERROR: Model testing failed."
    exit 1
fi

log "Pipeline completed successfully!"
log "Model saved at: $MODEL_DIR"
log "Test results saved at: test_results/"

# Print summary
echo ""
echo "=============================================="
echo "Vehicle Orientation Detection Pipeline Summary"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL_FILE"
echo "Test results: test_results/"
echo "Log file: $LOG_FILE"
echo "=============================================="