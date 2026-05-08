#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

GEMMA_MODEL="google/gemma-2-2b-it"
GEMMA_TAG="gemma2-2b-it"
GEMMA_LAYER=25

LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LLAMA_TAG="llama3.1-8b-it"
LLAMA_LAYER=11

run_pipeline () {
  local model_id="$1"
  local model_tag="$2"
  local extraction_layer="$3"

  echo "================================================================"
  echo " EDA  ($model_tag)"
  echo "================================================================"
  python scripts/run_eda.py --model "$model_id" --tag "$model_tag"

  echo "================================================================"
  echo " EXTRACT DIRECTIONS  ($model_tag, layer=$extraction_layer)"
  echo "================================================================"
  python scripts/extract_directions.py \
    --model "$model_id" \
    --tag "$model_tag" \
    --layer "$extraction_layer"

  echo "================================================================"
  echo " 2x2 MATRIX  ($model_tag)"
  echo "================================================================"
  python scripts/run_2x2_matrix.py \
    --model "$model_id" \
    --tag "$model_tag" \
    --layer "$extraction_layer"
  python scripts/plot_matrix.py --tag "$model_tag"

  echo "================================================================"
  echo " SENSITIVITY  ($model_tag)"
  echo "================================================================"

  local total_blocks
  total_blocks=$(python -c "
from transformers import AutoConfig
import os
from dotenv import load_dotenv
load_dotenv()
config = AutoConfig.from_pretrained('$model_id', token=os.getenv('HF_TOKEN'))
print(config.num_hidden_layers)
")
  local highest_valid_layer=$((total_blocks - 1))

  local lower_layer=$((extraction_layer - 3))
  local upper_layer=$((extraction_layer + 1))
  if [ "$upper_layer" -gt "$highest_valid_layer" ]; then
    upper_layer="$highest_valid_layer"
  fi

  local sweep_layers="$lower_layer,$((lower_layer + 1)),$((lower_layer + 2)),$extraction_layer"
  if [ "$upper_layer" -ne "$extraction_layer" ]; then
    sweep_layers="$sweep_layers,$upper_layer"
  fi

  python scripts/run_sensitivity.py \
    --model "$model_id" \
    --tag "$model_tag" \
    --layers "$sweep_layers" \
    --ranks 1,2,4,8,16 \
    --rank_layer "$extraction_layer"
}

case "$MODE" in
  gemma_only)
    run_pipeline "$GEMMA_MODEL" "$GEMMA_TAG" "$GEMMA_LAYER"
    ;;
  llama_only)
    run_pipeline "$LLAMA_MODEL" "$LLAMA_TAG" "$LLAMA_LAYER"
    ;;
  full|*)
    run_pipeline "$GEMMA_MODEL" "$GEMMA_TAG" "$GEMMA_LAYER"
    run_pipeline "$LLAMA_MODEL" "$LLAMA_TAG" "$LLAMA_LAYER"
    ;;
esac

echo "DONE — check results/<tag>/ for outputs"
