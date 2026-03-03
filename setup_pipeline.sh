#! /usr/bin/env bash

# Parameters
MODEL_NAME="lightgbm"

# Getting the project directory
PROJECT_DIR=$(dirname $(realpath $0))

# Retrieving the virtualenv binary directory
VENV_BIN=${PROJECT_DIR}/.venv/bin

# Activating the environment
echo ".. Activating virtual environment"
source ${VENV_BIN}/activate

# Setting up the dvc environment
DVC="${VENV_BIN}/dvc"
test -f "${DVC}" || (echo "dvc is not yet installed and configured. Use \"uv sync\"" && exit 1)

# Stage 1: Preparation
echo ".. Defining Stage 1: Split"
${DVC} stage add --force -n "split" \
  -d "src/data/data_split.py" \
  -d "data/raw_data/raw.csv" \
  -o "data/processed/X_train.csv" \
  -o "data/processed/X_test.csv" \
  -o "data/processed/y_train.csv" \
  -o "data/processed/y_test.csv" \
  python src/data/data_split.py

# Stage 2: Normalization
echo ".. Defining Stage 2: Normalize"
${DVC} stage add --force -n "normalize" \
  -d "src/data/normalize.py" \
  -d "data/processed/X_train.csv" \
  -d "data/processed/X_test.csv" \
  -o "data/processed/X_train_scaled.csv" \
  -o "data/processed/X_test_scaled.csv" \
  -o "models/scaler.joblib" \
  python src/data/normalize.py

# Stage 3: GridSearch
echo ".. Defining Stage 3: GridSearch"
${DVC} stage add --force -n "gridsearch" \
  -d "src/models/grid_search.py" \
  -d "src/models/registry.py" \
  -d "params.yaml" \
  -d "data/processed/X_train_scaled.csv" \
  -d "data/processed/y_train.csv" \
  -o "models/best_params.pkl" \
  -o "metrics/${MODEL_NAME}_results.csv" \
  python src/models/grid_search.py \
    --config "params.yaml" 

echo ".. Defining Stage 4: Training"
${DVC} stage add --force -n "training" \
  -d "src/models/training.py" \
  -d "src/models/registry.py" \
  -d "params.yaml" \
  -d "models/best_params.pkl" \
  -d "data/processed/X_train_scaled.csv" \
  -d "data/processed/y_train.csv" \
  -o "models/${MODEL_NAME}.pkl" \
  python src/models/training.py \
    --config "params.yaml" \
    --params "models/best_params.pkl" \
    --X_train_path "data/processed/X_train_scaled.csv" \
    --y_train_path "data/processed/y_train.csv" 

echo ".. Defining Stage 5: Evaluate"
${DVC} stage add --force -n "evaluate" \
  -d "src/models/evaluate.py" \
  -d "models/${MODEL_NAME}.pkl" \
  -d "data/processed/X_test_scaled.csv" \
  -d "data/processed/y_test.csv" \
  -o "data/prediction.csv" \
  -M metrics/scores.json \
  python src/models/evaluate.py \
    --model_path "models/${MODEL_NAME}.pkl" \
    --X_test_path "data/processed/X_test_scaled.csv" \
    --y_test_path "data/processed/y_test.csv" \
    --output "data/prediction.csv"
