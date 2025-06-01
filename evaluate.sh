#!/bin/bash

# Path to input images
input_dir="data"

# Loop through each results folder
for model_dir in results/*; do
  if [ -d "$model_dir" ]; then
    model_name=$(basename "$model_dir")
    
    # Assume the model file is the only .onnx file in the folder
    model_path=$(find "$model_dir" -name '*.onnx' | head -n 1)
    
    # Skip if no ONNX model found
    if [ -z "$model_path" ]; then
      echo "No ONNX model found in $model_dir"
      continue
    fi

    # Define output directory for evaluation results
    output_dir="${model_dir}/evaluation"
    mkdir -p "$output_dir"

    echo "Evaluating $model_name..."
    python ./development/evaluate.py "$input_dir" "$output_dir" "$model_dir"
  fi
done
