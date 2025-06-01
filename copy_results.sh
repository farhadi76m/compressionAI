#!/bin/bash

mkdir -p results/statics

for eval_dir in results/*/evaluation; do
  if [ -d "$eval_dir" ]; then
    summary_file="$eval_dir/results_summary.txt"
    if [ -f "$summary_file" ]; then
      # Extract model folder name, e.g. bmshj2018_factorized_q1
      model_name=$(basename $(dirname "$eval_dir"))
      cp "$summary_file" "results/statics/${model_name}.txt"
      echo "Copied $summary_file to results/statics/${model_name}.txt"
    else
      echo "No results_summary.txt in $eval_dir"
    fi
  fi
done



#!/bin/bash

output_csv="results/statics/summary.csv"

# Write CSV header
echo "model_name,quality,avg_BPP,avg_PSNR" > "$output_csv"

for file in results/statics/*.txt; do
  filename=$(basename "$file")
  base="${filename%.txt}"

  model_name="${base%_q*}"
  quality="${base##*_q}"

  # Extract Average BPP line, get the numeric value after colon and before comma
  avg_bpp=$(grep "Average BPP" "$file" | awk -F'[:,]' '{print $2}' | tr -d ' ')

  # Extract Average PSNR line, get the numeric value after "Average PSNR:" and before " dB"
  avg_psnr=$(grep "Average PSNR" "$file" | awk -F'[:,]' '{print $4}' | sed 's/ dB//g' | tr -d ' ')

  echo "${model_name},${quality},${avg_bpp},${avg_psnr}" >> "$output_csv"
done

echo "Summary CSV generated at $output_csv"

