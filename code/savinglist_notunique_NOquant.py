"""
Sample pixel grids from the raw movie and build the point list, as mentioned in Section 2.2 of paper.

For each of `r` randomly placed non-overlapping pixel grids, samples one
pixel per k-block, follows its trajectory across all timesteps of the movie,
and records the resulting point (one coordinate per timestep) plus
its originating pixel and grid index. Output is a checkpointed dict:

Usage:
  python newsaving_notunique_NOquant.py \\
      --input_data 1nM_24_hours_data.npy \\
      --group 0 --hours 6 --k 5 \\
      --output_file 24hmov_6hours_gr0_list_allpoints_k5_s6_r1000.npy
"""

import argparse
import os
import pickle
import tempfile

import numpy as np


def get_big_indices(data, maximum=7500):
    """Return [t, i, j] triples for pixels exceeding `maximum`"""
    big_timesteps_indices = []
    for t in range(data.shape[0]):
        indices = np.where(data[t, :, :] > maximum)
        if indices[0].size > 0:
            for i in range(indices[0].size):
                big_timesteps_indices.append([t, indices[0][i], indices[1][i]])
    return np.array(big_timesteps_indices)


def new_normalization(data):
    minimum = np.min(data)
    return (data - minimum) / 1000


def select_pixel_no_hotpoint(data, timestep, k, maximum=7500):
    """Pick one random pixel per k x k block, avoiding known hot pixels"""
    grid_size = 2 ** k
    selected_pixels = []
    selected_values = []
    hot_points = get_big_indices(data, maximum)
    hot_points = [tuple(x[1:]) for x in hot_points]
    hot_points = hot_points[0:2]
    for i in range(0, data.shape[1], grid_size):
        for j in range(0, data.shape[2], grid_size):
            pixel_range_i = np.arange(i, min(i + grid_size, data.shape[1]))
            pixel_range_j = np.arange(j, min(j + grid_size, data.shape[2]))
            selected_i = np.random.choice(pixel_range_i)
            selected_j = np.random.choice(pixel_range_j)
            while (selected_i, selected_j) in hot_points:
                selected_i = np.random.choice(pixel_range_i)
                selected_j = np.random.choice(pixel_range_j)
            selected_pixels.append((selected_i, selected_j))
            selected_values.append(data[timestep, selected_i, selected_j])
    return np.array(selected_pixels), np.array(selected_values)


def flatten_pixel_window_no_hotpoint(data, timestep, k, window):
    selected_pixel_2d, selected_values_2d_t = select_pixel_no_hotpoint(data, timestep=timestep, k=k)
    selected_values_all = selected_values_2d_t
    for t in range(timestep + 1, timestep + window):
        selected_values_t = data[t, selected_pixel_2d[:, 0], selected_pixel_2d[:, 1]]
        selected_values_all = np.concatenate((selected_values_all, selected_values_t))
    return selected_values_all, selected_pixel_2d


def atomic_save(save_file, obj):
    directory = os.path.dirname(os.path.abspath(save_file)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".npy")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            np.save(fh, obj, allow_pickle=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, save_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(save_file):
    """Handles missing and corrupt files"""
    empty = {"all_points": [], "all_pixels": [], "timesteps": [], "grid_indices": []}

    try:
        results = np.load(save_file, allow_pickle=True).item()
    except FileNotFoundError:
        print("No existing file found. Starting fresh.")
        return empty, set()
    except (EOFError, ValueError, OSError, pickle.UnpicklingError) as err:
        # A job killed mid-np.save leaves a truncated pickle. It is not
        # recoverable (np.save writes one blob), so park it and restart.
        parked = save_file + ".truncated"
        print(f"WARNING: {save_file} is unreadable ({type(err).__name__}: {err}).")
        print(f"         Moving it to {parked} and starting fresh.")
        os.replace(save_file, parked)
        return empty, set()

    used_pixels = {tuple(map(tuple, pixels)) for pixels in results["all_pixels"]}
    print(f"Loaded existing results from {save_file}. "
          f"Completed grids: {len(used_pixels)}, points so far: {len(results['all_points'])}")
    return results, used_pixels


def get_points_p_t_q_t_saving(data, timestep, k, window, s, r, save_file, save_every=10):
    norm_data = new_normalization(data)
    timesteps = np.arange(0, norm_data.shape[0] - window, 1)

    results, used_pixels = load_checkpoint(save_file)

    n_done = len(used_pixels)
    if n_done >= r:
        print(f"All {r} grids already present in {save_file}. Nothing to do.")
        return results["all_points"], results["all_pixels"], results["grid_indices"], timesteps
    if n_done > 0:
        print(f"Resuming: {n_done}/{r} grids done, {r - n_done} to go.")

    all_points = []
    all_pixels = []
    grid_indices = []

    for j in range(n_done, r):
        # Ensure unique pixel selection
        while True:
            _, selected_pixel_t = flatten_pixel_window_no_hotpoint(norm_data, timestep, k, window)
            selected_pixel_tuple = tuple(map(tuple, selected_pixel_t))
            if selected_pixel_tuple not in used_pixels:
                used_pixels.add(selected_pixel_tuple)
                break
            print(f"Duplicate pixels found for grid {j}, re-selecting...")

        for t in timesteps[1:]:
            selected_values_all = []
            for i in range(t, t + window, 1):
                selected_values_for_i = norm_data[i, selected_pixel_t[:, 0], selected_pixel_t[:, 1]]
                selected_values_all = np.concatenate((selected_values_all, selected_values_for_i))

            all_points.append(selected_values_all)
            all_pixels.append(selected_pixel_t)
            grid_indices.append(j)
            results["timesteps"].append(int(t))

        # Save results after every `save_every` grids
        if (j + 1) % save_every == 0 or j == r - 1:
            print(f"Saving results after processing grid {j + 1}/{r}", flush=True)
            results["all_points"].extend(all_points)
            results["all_pixels"].extend(all_pixels)
            results["grid_indices"].extend(grid_indices)
            atomic_save(save_file, results)
            print(f"  Saved {len(results['all_points'])} points "
                  f"({os.path.getsize(save_file) / 1e9:.2f} GB)", flush=True)

            all_points = []
            all_pixels = []
            grid_indices = []

    return results["all_points"], results["all_pixels"], results["grid_indices"], timesteps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_data", required=True,
                         help="Path to the raw movie .npy array, shape (timesteps, height, width).")
    parser.add_argument("--group", required=True,
                         help="Time-segment index to slice out of input_data (int, unused when --hours 24).")
    parser.add_argument("--hours", type=int, required=True, choices=[6, 12, 24],
                         help="Length in hours of the segment to analyze. 24 uses the full movie.")
    parser.add_argument("--k", type=int, required=True, help="Block-size exponent: pixel grid block is 2**k.")
    parser.add_argument("--output_file", required=True, help="Path to write the checkpointed point-list .npy to.")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--s", type=int, default=6)
    parser.add_argument("--r", type=int, default=1000, help="Number of independent pixel grids to sample.")
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    movie = np.load(args.input_data, allow_pickle=True)

    if args.hours in (6, 12):
        end = 720 * args.hours
        group = int(args.group)
        data = movie[group * end:end * (group + 1), :, :]
    else:  # 24
        data = movie

    print(f"Selected parameters k={args.k}, s={args.s}, w={args.window}, r={args.r}")
    print(f"Group {args.group}, hours {args.hours}, data shape {data.shape}")
    print(f"File to save: {args.output_file}")

    get_points_p_t_q_t_saving(
        data, timestep=0, k=args.k, window=args.window, s=args.s, r=args.r,
        save_file=args.output_file, save_every=args.save_every,
    )
