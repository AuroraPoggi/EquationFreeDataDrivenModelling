"""
Generate the tiny synthetic movie shipped in this directory to test the pipeline.

Usage:
  python generate_test_data.py
"""

import numpy as np

OUTPUT_FILE = "tiny_movie.npy"
TIMESTEPS = 12
HEIGHT = 512
WIDTH = 512
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    # Values well below the hot-pixel threshold (7500)
    movie = rng.uniform(low=0, high=300, size=(TIMESTEPS, HEIGHT, WIDTH)).astype(np.float32)
    np.save(OUTPUT_FILE, movie)
    print(f"Wrote {OUTPUT_FILE}: shape={movie.shape}, dtype={movie.dtype}, "
          f"size={movie.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
