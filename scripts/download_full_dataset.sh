#!/bin/bash
# Download the full dataset from Zenodo and place it under ./data.
#
# TODO before publishing: replace ZENODO_RECORD with the real record ID
# once the dataset has been uploaded and the record is public, and update
# the DOI referenced in ../README.md and ../CITATION.cff to match.
set -euo pipefail

ZENODO_RECORD="22159863"   # 10.5281/zenodo.22159863 
DATA_DIR="$(dirname "$0")/../data"

mkdir -p "${DATA_DIR}"

echo "Downloading dataset from Zenodo record ${ZENODO_RECORD} into ${DATA_DIR} ..."
zenodo_get "${ZENODO_RECORD}" -o "${DATA_DIR}"
# Alternative without the zenodo_get helper (pip install zenodo_get):
#   curl -L -o "${DATA_DIR}/1nM_24_hours_data.npy" \
#       "https://zenodo.org/record/${ZENODO_RECORD}/files/1nM_24_hours_data.npy"

echo "Done. Point --input_data / input_movie in your config at a file under ${DATA_DIR}."
