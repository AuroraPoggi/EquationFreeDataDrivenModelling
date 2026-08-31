"""
Compute the successor map function f over the point list, as mentioned in Section 2.3 of paper.

For every point recorded at the maximum timestep of the segment, finds its
nearest neighbour (excluding itself) among all points
below the halfway timestep threshold, via a KDTree.

Usage:
  python function_f_NOquant.py \\
      --group 0 --hours 6 --k 5 \\
      --input_file 24hmov_6hours_gr0_list_allpoints_k5_s6_r1000.npy \\
      --output_file 24hmov_6hours_gr0_k5_function_f_allpoints_r1000.npy
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.neighbors import KDTree


def opt_find_closest_elements_time_kdtree_smaller(y_all, X, timesteps, tree):
    """Nearest neighbour of y_all in X (excluding exact matches), via a prebuilt KDTree"""
    dist, ind = tree.query(np.array(y_all).reshape(1, -1), k=10)
    dist = dist.ravel()
    ind = ind.ravel()

    if len(ind) == 0:
        raise ValueError("KDTree returned no neighbors")

    X_arr = np.asarray(X)

    # Exclude neighbors that are exactly equal to the query point
    neighbor_points = X_arr[ind]
    equal_mask = np.all(neighbor_points == y_all, axis=1)
    filtered_dist = dist.copy()
    filtered_dist[equal_mask] = np.inf

    if np.all(np.isinf(filtered_dist)):
        # If all were equal to query, fall back to original distances (keep one)
        min_pos = 0
    else:
        min_pos = int(np.argmin(filtered_dist))

    chosen_idx = ind[min_pos]
    chosen_point = X_arr[chosen_idx]
    chosen_timestep = timesteps[chosen_idx]
    chosen_distance = dist[min_pos]
    return (chosen_point, chosen_timestep, chosen_distance)


def get_row_index(quant_points, closest_point, timesteps):
    """Row index (with the earliest timestep, in case of duplicates) of closest_point in quant_points"""
    closest_point_index = [
        idx for idx, point in enumerate(quant_points)
        if np.array_equal(closest_point[0], point)
    ]

    if not closest_point_index:
        print("No closest point found.")
        return None, None

    smallest_timestep_index = min(closest_point_index, key=lambda idx: timesteps[idx])
    smallest_timestep = timesteps[smallest_timestep_index]
    return smallest_timestep_index, smallest_timestep


def opt_function_f(npy_data, threshold, k, output_file_path, checkpoint_every=500):
    quant_points = np.asarray([p for p in npy_data['all_points']])
    timesteps = np.asarray(npy_data['timesteps'])
    total_points = len(quant_points)
    # print(f"Number of points in entire list {len(quant_points)}")

    max_timestep = timesteps.max()
    # print(f"Max timestep is {max_timestep}")

    # Build the KDTree only over points below the threshold timestep
    mask_below = timesteps < threshold
    X = quant_points[mask_below]
    # print(f"Number of points below threshold {len(X)}")
    X_timesteps = timesteps[mask_below]

    tree = None
    if len(X) > 0:
        tree = KDTree(X, metric='chebyshev')
        print("KDTree built for points below threshold.")

    checkpoint_path = output_file_path.replace('.npy', '_checkpoint.npy')
    results = []
    start_idx = 0

    if os.path.isfile(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, resuming...")
        try:
            loaded = np.load(checkpoint_path, allow_pickle=True)
            results = list(loaded.item()) if loaded.ndim == 0 else list(loaded)
            start_idx = len(results)
            print(f"Resuming from index {start_idx} / {total_points}")
        except (EOFError, ValueError, pickle.UnpicklingError) as e:
            print(f"Checkpoint file is corrupt or incomplete ({e}). Starting from scratch.")
            results = []
            start_idx = 0
            os.remove(checkpoint_path)
    else:
        print("No checkpoint found, starting from scratch.")

    # Only run the expensive KD query for points at the max timestep;
    # everything else keeps its own index
    for idx in range(start_idx, total_points):
        pt = quant_points[idx]
        t = timesteps[idx]
        result = {'index_under_analysis': idx + 1, 'timestep': int(t)}

        if t == max_timestep and tree is not None:
            if idx % 10 == 0:
                print(f"Processing index {idx} at max timestep {t}")

            closest_element = opt_find_closest_elements_time_kdtree_smaller(pt, X, X_timesteps, tree)
            result['index_under_analysis'], result['timestep'] = get_row_index(quant_points, closest_element, timesteps)

        results.append(result)
        if (idx + 1) % checkpoint_every == 0:
            tmp_path = checkpoint_path.replace('.npy', '_tmp.npy')
            np.save(tmp_path, np.array(results, dtype=object))
            os.replace(tmp_path, checkpoint_path)
            print(f"Checkpoint saved at index {idx + 1} / {total_points}")

    tmp_output = output_file_path.replace('.npy', '_tmp.npy')
    np.save(tmp_output, results)
    os.replace(tmp_output, output_file_path)
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed: {checkpoint_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True)
    parser.add_argument("--hours", type=int, required=True, choices=[6, 12, 24])
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--input_file", required=True, help="Output of newsaving_notunique_NOquant.py.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--r_grids", type=int, default=1000, help="Number of grids r used to slice the point list.")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Group {args.group}, hours {args.hours}, k {args.k}")

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    npy_data = np.load(args.input_file, allow_pickle=True).item()
    print(f"The number of points in the entire list is {len(npy_data['all_points'])}")

    n_points = 720 * args.hours * args.r_grids
    npy_data = {key: v[:n_points] if isinstance(v, list) else v for key, v in npy_data.items()}
    print(f"The number of points after slicing is {len(npy_data['all_points'])}")

    threshold = int(max(npy_data['timesteps']) / 2)
    print(f"Threshold set to {threshold}")

    opt_function_f(npy_data, threshold, args.k, args.output_file, checkpoint_every=args.checkpoint_every)
