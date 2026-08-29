# Small test dataset

`tiny_movie.npy` is a small synthetic array (200 timesteps x 8x8 pixels, uniform
random noise - see `generate_test_data.py`) shipped with this repo so
the full pipeline in [`../code`](../code) can be run end-to-end without
downloading the real dataset from Zenodo (see
[`../scripts/download_full_dataset.sh`](../scripts/download_full_dataset.sh)).

Run [`../scripts/run_pipeline_example.sh`](../scripts/run_pipeline_example.sh)
to execute the full pipeline against this file.
