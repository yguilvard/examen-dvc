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
echo ".. Defining Stage 1: Prepare"
${DVC} stage add --force -n "prepare" \
  -d "src/data/prepare.py" \
  -d "data/raw_data/raw.csv" \
  -o "data/processed/X_train.csv" \
  -o "data/processed/X_test.csv" \
  -o "data/processed/y_train.csv" \
  -o "data/processed/y_test.csv" \
  python src/data/prepare.py

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
  -d "src/models/gridsearch.py" \
  -d "src/models/registry.py" \
  -d "configs/gridsearch_${MODEL_NAME}.json" \
  -d "data/processed/X_train_scaled.csv" \
  -d "data/processed/y_train.csv" \
  -o "models/${MODEL_NAME}-best-params.pkl" \
  -o "metrics/${MODEL_NAME}_results.csv" \
  python src/models/gridsearch.py \
    --config "configs/gridsearch_${MODEL_NAME}.json" 

echo ".. Defining Stage 4: Training"
${DVC} stage add --force -n "training" \
  -d "src/models/train.py" \
  -d "src/models/registry.py" \
  -d "configs/gridsearch_${MODEL_NAME}.json" \
  -d "models/${MODEL_NAME}-best-params.pkl" \
  -d "data/processed/X_train_scaled.csv" \
  -d "data/processed/y_train.csv" \
  -o "models/${MODEL_NAME}.pkl" \
  python src/models/train.py \
    --config "configs/gridsearch_${MODEL_NAME}.json" \
    --params "models/${MODEL_NAME}-best-params.pkl" \
    --X_train_path "data/processed/X_train_scaled.csv" \
    --y_train_path "data/processed/y_train.csv" 

echo ".. Defining Stage 5: Evaluate"
${DVC} stage add --force -n "evaluate" \
  -d "src/models/evaluate.py" \
  -d "models/${MODEL_NAME}.pkl" \
  -d "data/processed/X_test_scaled.csv" \
  -d "data/processed/y_test.csv" \
  -o "data/predictions.csv" \
  -M metrics/scores.json \
  python src/models/evaluate.py \
    --model_path "models/${MODEL_NAME}.pkl" \
    --X_test_path "data/processed/X_test_scaled.csv" \
    --y_test_path "data/processed/y_test.csv"
