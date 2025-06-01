#!/bin/bash

# Correct model names with hyphens
models=(
    bmshj2018-factorized
    bmshj2018-factorized-relu
    bmshj2018-hyperprior
    cheng2020-anchor
    cheng2020-attn
    mbt2018
    mbt2018-mean
)

qualities=(1 2 3 4)

mkdir -p results

for model in "${models[@]}"; do
  for quality in "${qualities[@]}"; do
    # Replace hyphens with underscores for folder names
    model_safe=${model//-/_}
    out_dir="results/${model_safe}_q${quality}"
    mkdir -p "$out_dir"
    python ./development/convert_onnx.py \
      --input ./data/kodim01.png \
      --export-dir "$out_dir" \
      --model-type "$model" \
      --quality "$quality"
  done
done
