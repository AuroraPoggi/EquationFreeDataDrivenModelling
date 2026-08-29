#!/bin/bash
# Run the full 7-step pipeline locally, end to end, against the small test
# dataset shipped in ../small_test_dataset. 
#
# To run on the full dataset instead: 
# run scripts/download_full_dataset.sh
# then override the variables below to point at the downloaded file 
# and a larger R_GRIDS/pairs list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${REPO_ROOT}/code"

# ── run parameters (override via environment, e.g. `GROUP=1 ./run_pipeline_example.sh`) ──
GROUP="${GROUP:-0}"
HOURS="${HOURS:-6}"
# K_GRID=8 matches small_test_dataset/tiny_movie.npy (512x512 frames): 
K_GRID="${K_GRID:-8}"
R_GRIDS="${R_GRIDS:-100}"
DATE_TAG="${DATE_TAG:-testrun}"

INPUT_MOVIE="${INPUT_MOVIE:-${REPO_ROOT}/small_test_dataset/tiny_movie.npy}"
PAIRS_FILE="${PAIRS_FILE:-${REPO_ROOT}/small_test_dataset/pairs_list_tiny.txt}"
OUTBASE="${OUTBASE:-${REPO_ROOT}/pipeline_output}"

BASE="${OUTBASE}/gr${GROUP}_r${R_GRIDS}_${DATE_TAG}"
STAGE1_DIR="${BASE}/stage1_bestcycles"
STAGE2_DIR="${BASE}/stage2_plateau"
STAGE3_DIR="${BASE}/stage3_correlations"
STAGE4_DIR="${STAGE3_DIR}/eigenvalues"
mkdir -p "${BASE}" "${STAGE1_DIR}" "${STAGE2_DIR}" "${STAGE3_DIR}" "${STAGE4_DIR}"

FILE_LIST="${BASE}/list_allpoints_gr${GROUP}_k${K_GRID}_r${R_GRIDS}_${DATE_TAG}.npy"
FILE_FUNCF="${BASE}/function_f_gr${GROUP}_k${K_GRID}_r${R_GRIDS}_${DATE_TAG}.npy"
FILE_CYC="${BASE}/unique_cycles_gr${GROUP}_k${K_GRID}_r${R_GRIDS}_${DATE_TAG}.npy"
FILE_CYC_CSV="${BASE}/unique_cycles_gr${GROUP}_k${K_GRID}_r${R_GRIDS}_${DATE_TAG}.csv"

PAIRS_ARGS=()
while IFS= read -r line; do
    clean="$(echo "${line}" | tr -d '\r\n\t ')"
    [[ -n "${clean}" && "${clean}" != \#* ]] && PAIRS_ARGS+=("${clean}")
done < "${PAIRS_FILE}"

echo "── Step 1: sample point list from the raw movie ──"
python3 -u "${CODE}/savinglist_notunique_NOquant.py" \
    --input_data "${INPUT_MOVIE}" --group "${GROUP}" --hours "${HOURS}" --k "${K_GRID}" \
    --output_file "${FILE_LIST}" --r "${R_GRIDS}"

echo "── Step 2: build the return-map function f ──"
python3 -u "${CODE}/function_f_NOquant.py" \
    --group "${GROUP}" --hours "${HOURS}" --k "${K_GRID}" \
    --input_file "${FILE_LIST}" --output_file "${FILE_FUNCF}" --r_grids "${R_GRIDS}"

echo "── Step 3: trace cycles and count basin sizes ──"
python3 -u "${CODE}/getcycles_countpointsinbasin.py" \
    --group "${GROUP}" --k "${K_GRID}" \
    --input_file "${FILE_FUNCF}" --output_file "${FILE_CYC}" --output_file_csv "${FILE_CYC_CSV}"

echo "── Stage 1: mark all valid cycles as good, per projection pair ──"
python3 -u "${CODE}/pipeline_stage1_bestcycles.py" \
    --file_cyc "${FILE_CYC}" --pairs_file "${PAIRS_FILE}" \
    --stage1_dir "${STAGE1_DIR}" --r_grids "${R_GRIDS}"

N_CYCLES=$(python3 -c "
import numpy as np
cycles = np.load('${FILE_CYC}', allow_pickle=True)
print(sum(1 for c in cycles if len(c) > 1))
")
echo "N_CYCLES=${N_CYCLES}"

echo "── Stage 2: entropy / plateau detection ──"
python3 -u "${CODE}/pipeline_stage2_plateau.py" \
    --group "${GROUP}" --hours "${HOURS}" --k_grid "${K_GRID}" \
    --file_f "${FILE_LIST}" --file_cyc "${FILE_CYC}" \
    --stage1_dir "${STAGE1_DIR}" --output_dir "${STAGE2_DIR}" \
    --pairs "${PAIRS_ARGS[@]}" --n_cycles "${N_CYCLES}" --r_grids "${R_GRIDS}" --all_cycles_good true \
    --deltas 0.10 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.7 0.8 1.0

echo "── Stage 3: cross-correlations ──"
python3 -u "${CODE}/pipeline_stage3_correlations.py" \
    --group "${GROUP}" --hours "${HOURS}" --k_grid "${K_GRID}" \
    --file_f "${FILE_LIST}" --file_cyc "${FILE_CYC}" \
    --stage1_dir "${STAGE1_DIR}" --stage2_dir "${STAGE2_DIR}" --output_dir "${STAGE3_DIR}" \
    --pairs "${PAIRS_ARGS[@]}" --M_values "${N_CYCLES}" --abs_max_delay 100 --zero_tol 1e-8 --zero_run 100 \
    --r_grids "${R_GRIDS}" --all_cycles_good true

echo "── Stage 4: eigenvalues ──"
python3 -u "${CODE}/pipeline_stage4_eigenvalues.py" \
    --group "${GROUP}" --hours "${HOURS}" --k_grid "${K_GRID}" \
    --file_f "${FILE_LIST}" --file_cyc "${FILE_CYC}" \
    --stage1_dir "${STAGE1_DIR}" --stage2_dir "${STAGE2_DIR}" --output_dir "${STAGE4_DIR}" \
    --pairs "${PAIRS_ARGS[@]}" --M_values "${N_CYCLES}" --r_grids "${R_GRIDS}" --all_cycles_good true

echo ""
echo "Pipeline complete. Outputs under ${BASE}"
