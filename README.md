# Equation Free Data Driven Modelling

Code of paper "Equation Free Data Driven Modelling of chaotic processes."

We introduce a method for constructing predictive models of non-cyclic physical processes directly from time-series data, without assuming an underlying differential equation. The observations define a discrete evolution rule whose recurrent behaviour captures the essential dynamics of the process. Analysing this behaviour across multiple geometric scales leads to probabilistic models in the form of Markov chains. Hyperbolicity criteria identify when these models provide a consistent statistical description of the data. The method is inspired by the analysis of a biological imaging data set referred to as the Cell Process in the paper.


The full movie dataset used in the paper is **hosted on Zenodo**,
not in this repository (see [Getting the full dataset](#getting-the-full-dataset)
below). This repo has a tiny synthetic dataset so the pipeline can be run
end-to-end on a laptop without downloading anything, however, it does not have any meaning the example.

## Repository layout

```
.
├── code/            7 pipeline scripts (see "Pipeline steps" below)
├── configuration/   Example config + real projection-pair list used in the paper
├── small_test_dataset/  Synthetic movie + generator, for a quick test
├── scripts/         run_pipeline_example.sh, download_full_dataset.sh
├── requirements.txt
├── CITATION.cff
└── LICENSE (MIT)
```

This is the code + small-test-data + instructions half of a two-part release;
the other half (the full dataset) lives on Zenodo:

```
GitHub (this repo)                          Zenodo
├── code                                     ├── full dataset 
├── configuration                DOI  ──►    ├── metadata
├── small_test_dataset                       └── version
└── scripts (download/run)
```

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+.

## Quick start: run the pipeline on the small test dataset

```bash
./scripts/run_pipeline_example.sh
```

This runs the full 7-step pipeline (described in the paper) against
[`small_test_dataset/tiny_movie.npy`](small_test_dataset/tiny_movie.npy), a
small synthetic array shipped in this repo (see
[`small_test_dataset/README.md`](small_test_dataset/README.md) — **it is not
real experimental data**, and the cycles/entropies/correlations it produces
are not scientifically meaningful). Outputs are written under
`pipeline_output/`.

Run parameters can be overridden via environment variables, e.g.:

```bash
GROUP=1 R_GRIDS=200 ./scripts/run_pipeline_example.sh
```

See [`configuration/example_config.yaml`](configuration/example_config.yaml)
for a documented list of all parameters.

## Pipeline steps

The pipeline consists of 3 data-preparation steps followed by 4 analysis
stages. Each step reads the previous step's output file.

| # | Script | Purpose |
|---|--------|---------|
| Step 1 | [`code/savinglist_notunique_NOquant.py`](code/savinglist_notunique_NOquant.py) | Samples a point list from the raw movie: for each of `r_grids` random pixel grids, extracts a block of pixels and follows it forward in time. |
| Step 2 | [`code/function_f_NOquant.py`](code/function_f_NOquant.py) | Builds the discrete return map `f` over the sampled points, using a KD-tree nearest-neighbor search to find each point's closest successor. |
| Step 3 | [`code/getcycles_countpointsinbasin.py`](code/getcycles_countpointsinbasin.py) | Traces cycles in `f`, strips basin (non-periodic) parts, merges overlapping cycles, and writes the resulting unique cycles (`.npy` + `.csv`). 
*** Note: set at the beginning of the scrip the variable: MOVIE_LENGTH = 4318 ***|
| Stage 1 | [`code/pipeline_stage1_bestcycles.py`](code/pipeline_stage1_bestcycles.py) | For each projection pair in `--pairs_file`, marks every valid (period > 1) cycle as "good". |
| Stage 2 | [`code/pipeline_stage2_plateaux_newnewnew.py`](code/pipeline_stage2_plateaux_newnewnew.py) | Computes per-cycle and joint entropies over a range of renormalization scales `k`, and detects the entropy plateau (`k*`). |
| Stage 3 | [`code/pipeline_stage3_correlations.py`](code/pipeline_stage3_correlations.py) | Builds the transition matrix `P_k` at `k*` and computes cross-correlation functions over a range of time delays for each projection pair. |
| Stage 4 | [`code/pipeline_stage4_eigenvalues.py`](code/pipeline_stage4_eigenvalues.py) | Computes the leading eigenvalues of `P_k` at each `(M, k*)` combination. |



## Getting the full dataset

The full dataset (real experimental movies) used to produce the
paper's results is hosted on Zenodo:

```bash
pip install zenodo_get
./scripts/download_full_dataset.sh
```


Once downloaded, point `INPUT_MOVIE` at the downloaded file and
`PAIRS_FILE` at [`configuration/pairs_list_example.txt`](configuration/pairs_list_example.txt)
(the real projection pairs used in the paper — sized for the full dataset's
point dimensionality, unlike the tiny pairs file used by the quick-start
above), then re-run `scripts/run_pipeline_example.sh` with a larger
`K_GRID`/`R_GRIDS` matching your data.

## Citing

See [`CITATION.cff`](CITATION.cff). If you use this code or the accompanying
dataset, please cite both the paper and the Zenodo dataset record.

## License

MIT — see [`LICENSE`](LICENSE).
