"""
Trace cycles in the return map f and count basin sizes.

Starting from each movie's first point, follows the return map f until a repeated value is hit, giving one
raw "cycle with basin tail" per movie. The basin tail (the transient before
the point sequence starts repeating) is then stripped, and cycles that share
at least one point are merged, with the number of basin points funneling
into each surviving cycle tracked and summed.


Usage:
  python main_getcycles_countpointsinbasin.py \\
      --group 0 --k 5 \\
      --input_file 24hmov_6hours_gr0_k5_function_f_allpoints_r1000.npy \\
      --output_file 24hmov_6hours_gr0_unique_cycles_basinmovies_period_k5_s6_r1000.npy \\
      --output_file_csv 24hmov_6hours_gr0_uniquecycles_basinmovies_period_jumps_k5_s6_r1000.csv
"""

import argparse

import numpy as np
import pandas as pd

MOVIE_LENGTH = 358  # points per movie in the return-map sequence


def trace_cycles_over_starts(function_f, times):
    n = len(function_f)
    starts = list(range(0, len(function_f), MOVIE_LENGTH))

    cycles = []
    for start in starts:
        if start >= n:
            continue
        visited_values = set()
        cycle_indices = []
        cycle = []
        timesteps = []
        current_index = start
        while 0 <= current_index < n:
            value = function_f[current_index]
            if value in visited_values:
                # Append the repeated value and its info before breaking
                cycle_indices.append(current_index)
                cycle.append(value)
                timesteps.append(times[current_index])
                break
            visited_values.add(value)
            cycle_indices.append(current_index)
            cycle.append(value)
            timesteps.append(times[current_index])
            next_index = value
            if not isinstance(next_index, (int, np.integer)):
                try:
                    next_index = int(next_index)
                except Exception:
                    break
            current_index = next_index
        if cycle_indices:
            cycles.append({
                'cycle_indices': cycle_indices,
                'cycle': cycle,
                'timesteps': timesteps
            })
    return pd.DataFrame(cycles)


def remove_basin_parts(cycles_df):
    """Removes the basin parts of the cycles"""
    filtered_cycles = []
    for i in range(len(cycles_df)):
        value = cycles_df.iloc[i]['cycle'][-1]
        cycle_indices = cycles_df.iloc[i]['cycle_indices']
        timesteps = cycles_df.iloc[i]['timesteps']

        indices_in_cycle = [j for j, v in enumerate(cycles_df.iloc[i]['cycle']) if v == value]
        if indices_in_cycle[0] < len(cycles_df.iloc[i]['cycle']):
            first_index = indices_in_cycle[0]
            filtered_cycles.append({
                'cycle': cycles_df.iloc[i]['cycle'][first_index:],
                'cycle_indices': cycle_indices[first_index:],
                'timesteps': timesteps[first_index:]
            })

    return pd.DataFrame(filtered_cycles)


def remove_overlapping_cycles_pairwise_with_tracking(cycles_withoutbasin):
    """
    Removes cycles that overlap (share at least one point). When a cycle is dropped, its
    basin-point count is added to the surviving cycle it collided with
    """
    filtered_cycles = pd.DataFrame(columns=['cycle_nobasin', 'points_in_basin', 'period'])

    basin_counts = np.ones(len(cycles_withoutbasin), dtype=int)
    results = pd.DataFrame(columns=['cycle_nobasin', 'points_in_basin'])
    results['cycle_nobasin'] = cycles_withoutbasin['cycle']
    results['points_in_basin'] = basin_counts

    keep_mask = np.ones(len(results['cycle_nobasin']), dtype=bool)
    for i in range(len(results['cycle_nobasin'])):
        if not keep_mask[i]:
            continue
        current_cycle = set(results.iloc[i]['cycle_nobasin'])
        for j in range(i + 1, len(results['cycle_nobasin'])):
            if not keep_mask[j]:
                continue
            other_cycle = set(results.iloc[j]['cycle_nobasin'])
            if current_cycle & other_cycle:
                results.at[i, 'points_in_basin'] += results.iloc[j]['points_in_basin']
                keep_mask[j] = False

    filtered_cycles[['cycle_nobasin', 'points_in_basin']] = results[keep_mask][['cycle_nobasin', 'points_in_basin']].values
    filtered_cycles['period'] = filtered_cycles['cycle_nobasin'].apply(len)

    return filtered_cycles


def count_jumps_in_cycle(cycle):
    """Number of non-consecutive steps in a cycle (where next value != previous + 1)"""
    return sum(1 for i in range(1, len(cycle)) if cycle[i] != cycle[i - 1] + 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--input_file", required=True, help="Output of faster_function_f_NOquant.py.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_file_csv", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_f = np.load(args.input_file, allow_pickle=True)
    # print(f"Loaded data from {args.input_file} with {len(data_f)} entries.")
    function_f = np.array([item['index_under_analysis'] for item in data_f])
    times = np.array([item['timestep'] for item in data_f])

    cycles = trace_cycles_over_starts(function_f, times)
    # print(f"Cycle length: {len(cycles)}")
    # for i in range(len(cycles)):
    #     print(f"Cycle {i}: Length {len(cycles.iloc[i]['cycle'])}")

    cycles_withoutbasin = remove_basin_parts(cycles)
    # print(f"Cycles without basin parts: {len(cycles_withoutbasin)}")
    # for i in range(len(cycles_withoutbasin)):
    #     print(f"Cycle without basin {i}: Length {len(cycles_withoutbasin.iloc[i]['cycle'])}")

    unique_cycles = remove_overlapping_cycles_pairwise_with_tracking(cycles_withoutbasin)
    unique_cycles_order = unique_cycles.sort_values(by='points_in_basin', ascending=False).reset_index(drop=True)
    # print(f"Unique cycles after removing overlaps: {len(unique_cycles_order)}")

    unique_cycles_order_numpy = unique_cycles_order['cycle_nobasin'].to_numpy()
    try:
        np.save(args.output_file, unique_cycles_order_numpy)
    except PermissionError:
        print(f"Permission denied when trying to save to {args.output_file}")
        print("Check directory permissions or choose a different output path.")
        raise
    except OSError as e:
        print(f"OS error when saving to {args.output_file}: {e}")
        raise

    unique_cycles_order['num_jumps'] = unique_cycles_order['cycle_nobasin'].apply(count_jumps_in_cycle)
    unique_cycles_order.to_csv(args.output_file_csv, index=False)
