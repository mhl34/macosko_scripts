#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$1"

# Paths to external scripts and resources
RSCRIPT_DE="run-nemesh-de.R"
RSCRIPT_TRADE="run-trade-dream-annot.R"
GENESET_QS="~/zonation/TWI_analysis/gene_sets/D1_all_gsea_terms_with_block_and_disease.qs"
DIR_LABEL="paired_permutation_Astrocyte_DV_label_1000"

find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -n 1 -P 50 bash -c '
      dir="$1"
      name="$(basename "$dir")"
      out_dir="$dir/out"
      model_file="$out_dir/${name}_age_dream_model.qs"

      # --- STEP 00: Differential Expression ---
      run_de=true
      if [[ -d "$out_dir" ]]; then
        shopt -s nullglob
        qs_files=("$out_dir"/*.qs)
        shopt -u nullglob
        
        if (( ${#qs_files[@]} > 0 )); then
          run_de=false
          echo "[DE] Skipping $name (files exist)"
        fi
      fi

      if [[ "$run_de" == true ]]; then
        echo "[DE] Running on $name"
        Rscript "'"$RSCRIPT_DE"'" -d "'"$DIR_LABEL"'" -l "$name" -n 2
      fi

      # --- STEP 01: Trade / GSEA ---
      # Only run if the output directory exists (created by Step 00)
      if [[ -d "$out_dir" ]]; then
        if [[ -f "$model_file" ]]; then
            echo "[TRADE] Running on $name"
            Rscript "'"$RSCRIPT_TRADE"'" \
              -p "$model_file" \
              -n 2 \
              -g "'"$GENESET_QS"'" \
              -r age
        else
            echo "[TRADE] Warning: $model_file not found. Skipping."
        fi
      else
        echo "[TRADE] Skipping $name (out/ directory missing)"
      fi
    ' _
