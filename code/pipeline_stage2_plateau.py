"""
Entropy computation and plateau detection per (projection pair, cycle set), as introduced in the paper Section 2.10.
For each projection pair and each cycle, computes entropies_cycles over all renormalization sets U_k
and applies compute_plateau to locate the plateau scale k*.


Usage:
  python pipeline_stage2_plateaux_newnewnew.py \\
      --group 0 --hours 6 --k_grid 5 \\
      --file_f     <allpoints.npy> \\
      --file_cyc   <cycles.npy> \\
      --stage1_dir <stage1_output_dir> \\
      --output_dir <stage2_output_dir> \\
      --pairs "113,97" "97,113" "152,153" \\
      [--n_cycles 30] \\
      [--deltas 0.10 0.20 0.25 0.30 0.35] \\
      [--r_grids 1000] \\
      [--all_cycles_good false]
"""

import argparse
import json
import os
import sys
import csv
from scipy.linalg import eig
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import Counter

import numpy as np

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
]

def get_cycle_points(cycles, data_f):
    result = []
    all_points_ref = data_f['all_points']
    for i in range(len(cycles)):
        unique_cycle_indices = cycles[i]
        if len(unique_cycle_indices) == 1:
            continue
        arr = np.stack([all_points_ref[idx] for idx in unique_cycle_indices])
        result.append(arr)
    return result


def compute_uk(points, k_grid):
    U_k_all_dims = [points]
    for s in range(1500):
        current_points = U_k_all_dims[s]
        if len(current_points) <= 1:
            U_k_all_dims.append(current_points)
            continue
        dimensions = 2 ** (2 * (9 - k_grid))
        furthest_points = set()
        for dim in range(dimensions):
            dim_values = current_points[:, dim]
            furthest_points.update(np.where(dim_values == np.max(dim_values))[0])
            furthest_points.update(np.where(dim_values == np.min(dim_values))[0])
        keep_mask = np.ones(len(current_points), dtype=bool)
        keep_mask[list(furthest_points)] = False
        new_points = current_points[keep_mask]
        if len(new_points) == 0:
            center = np.mean(current_points, axis=0)
            distances = np.linalg.norm(current_points - center, axis=1)
            new_points = current_points[np.argmin(distances)].reshape(1, -1)
        U_k_all_dims.append(new_points)
    return U_k_all_dims


def singleton(U_k_all_dims):
    for idx, u_k in enumerate(U_k_all_dims):
        if u_k.shape[0] == 1:
            return idx
    return len(U_k_all_dims) - 1


def to_hashable_tuple(point):
    arr = np.asarray(point)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return tuple(arr.tolist())



########################## entropy computation ##########################

def entropies_cycles(tested_cyclepoints, periods, U_k_alldims,
                             cycle_ids, first_singleton_index):

    cycle_map = {cid: (tested_cyclepoints[cid], periods[idx])
                 for idx, cid in enumerate(cycle_ids)}
    num_cycles = len(cycle_ids)
    entropies = {'U_k': {}}

    for k in range(first_singleton_index + 1):
        uk_set = set(map(tuple, U_k_alldims[k]))
        N_gamma = {}
        F_gamma = {}
        E_gamma = {}

        for cid in cycle_ids:
            cycle_points, _ = cycle_map[cid]
            N, E, F = 0, 0, 0
            n = len(cycle_points)
            for t in range(n):
                pt_in      = tuple(cycle_points[t]) in uk_set
                pt_next_in = tuple(cycle_points[(t + 1) % n]) in uk_set
                if pt_in:
                    N += 1
                    if pt_next_in:
                        F += 1
                    else:
                        E += 1
            N_gamma[cid] = N
            F_gamma[cid] = F
            E_gamma[cid] = E

        sum_N = sum(N_gamma[cid] / cycle_map[cid][1] for cid in cycle_ids)
        sum_F = sum(F_gamma[cid] / cycle_map[cid][1] for cid in cycle_ids)

        sum1 = sum2 = 0.0
        for cid in cycle_ids:
            _, period = cycle_map[cid]
            n_term = N_gamma[cid] / period
            f_term = F_gamma[cid] / period
            if n_term > 0 and sum_N > 0:
                sum1 += n_term * np.log(period * sum_N)
            if f_term > 0 and sum_F > 0:
                sum2 += f_term * np.log(period * sum_F)

        entropy_total = (1 / num_cycles) * sum1 - (1 / num_cycles) * sum2
        entropies['U_k'][k] = {
            'entropy': float(entropy_total),
            'N_gamma': N_gamma,
            'F_gamma': F_gamma,
            'E_gamma': E_gamma,
        }

        if k % 100 == 0:
            print(f"    k={k}: h={entropy_total:.4f}")

    return entropies


########################## plateau computation ##########################

def compute_plateau(entropies_per_cycle, cycle_ids, first_singleton_index):
    k_values = list(range(first_singleton_index + 1))
    h_per_cycle = {cid: np.zeros(len(k_values)) for cid in cycle_ids}
    print(f"h per cycle is {h_per_cycle}, should be with max around 1.6/7")

    for k in k_values:
        for cid in cycle_ids:
            h_per_cycle[cid][k] = float(entropies_per_cycle[cid]['U_k'][k]['entropy'])

    # Verify condition 1: h(0)=0 and h(m)=0 for all cycles
    for cid in cycle_ids:
        h0 = h_per_cycle[cid][0]
        hm = h_per_cycle[cid][first_singleton_index]
        if abs(h0) > 1e-10 or abs(hm) > 1e-10:
            print(f"    WARNING: cycle {cid} h(0)={h0:.6f}, h(m)={hm:.6f} "
                  f"(should both be 0)")

    def get_S_gamma(cid, delta):
        h      = h_per_cycle[cid]
        thresh = float(np.max(h)) - delta
        return [k for k in k_values if h[k] > thresh]

    def is_unimodal_interval(k_list, cid):
        if not k_list:
            return False
        # Must be contiguous
        if k_list != list(range(k_list[0], k_list[-1] + 1)):
            return False
        h = h_per_cycle[cid]
        # Increasing up to k(0)
        for k in range(1, k_list[0] + 1):
            if h[k] < h[k-1]:
                return False
        # Decreasing from k(1)
        for k in range(k_list[-1], first_singleton_index):
            if h[k+1] > h[k]:
                return False
        return True


    # Collect ALL critical delta values:
    # 1. max_h_cid - h_cid(k) for each cid and k  (threshold crossings for S_gamma)
    # 2. |h_i(k) - h_j(k)| for each pair and k     (condition 3 crossings)
    candidates = set()
    for cid in cycle_ids:
        h     = h_per_cycle[cid]
        max_h = float(np.max(h))
        for k in k_values:
            candidates.add(max_h - float(h[k]))
    for a in range(len(cycle_ids)):
        for b in range(a + 1, len(cycle_ids)):
            ci, cj = cycle_ids[a], cycle_ids[b]
            for k in k_values:
                candidates.add(abs(float(h_per_cycle[ci][k])
                                   - float(h_per_cycle[cj][k])))

    # Test just above each critical value 
    test_deltas = sorted(c + 1e-9 for c in candidates if c >= 0)

    best_delta  = None
    best_result = None

    for delta in test_deltas:
        if delta <= 0:
            continue

        # Condition 1+2: each S_gamma must be a proper unimodal interval
        K_gamma = {}
        valid   = True
        for cid in cycle_ids:
            S_gamma = get_S_gamma(cid, delta)
            if not is_unimodal_interval(S_gamma, cid):
                valid = False
                break
            K_gamma[cid] = S_gamma

        if not valid:
            continue

        # S = intersection must be non-empty
        K_sets         = [set(K_gamma[cid]) for cid in cycle_ids]
        K_intersection = sorted(set.intersection(*K_sets))
        if not K_intersection:
            continue

        # Condition 3: max |h_i(k) - h_j(k)| < delta for all k in S
        max_diff = 0.0
        for a in range(len(cycle_ids)):
            for b in range(a + 1, len(cycle_ids)):
                ci, cj = cycle_ids[a], cycle_ids[b]
                for k in K_intersection:
                    diff = abs(float(h_per_cycle[ci][k])
                               - float(h_per_cycle[cj][k]))
                    if diff > max_diff:
                        max_diff = diff

        if max_diff >= delta:
            continue

        best_delta  = delta
        best_result = (K_gamma, K_intersection, max_diff)
        break

    if best_result is None:
        print(f"    ⚠ ALERT: No valid delta_III found satisfying all conditions.")
        return {
            'h_per_cycle':         {cid: h_per_cycle[cid].tolist() for cid in cycle_ids},
            'K_gamma':             {cid: [] for cid in cycle_ids},
            'k0_gamma':            {cid: None for cid in cycle_ids},
            'k1_gamma':            {cid: None for cid in cycle_ids},
            'K_intersection':      [],
            'k_star':              None,
            'delta_III':           None,
            'global_min_delta_III': None,
            'criterion_satisfied': False,
        }

    K_gamma, K_intersection, max_diff = best_result
    k_star   = K_intersection[len(K_intersection) // 2]
    k0_gamma = {cid: K_gamma[cid][0]  if K_gamma[cid] else None for cid in cycle_ids}
    k1_gamma = {cid: K_gamma[cid][-1] if K_gamma[cid] else None for cid in cycle_ids}

    print(f"    delta_III={best_delta:.6f}: "
          f"K_intersection=[{K_intersection[0]},{K_intersection[-1]}] "
          f"(size={len(K_intersection)}), k*={k_star}, "
          f"max|h_i-h_j| over S = {max_diff:.6f} < delta_III ✓")
    for cid in cycle_ids:
        print(f"      cycle {cid}: S_gamma=[{k0_gamma[cid]},{k1_gamma[cid]}], "
              f"max_h={float(np.max(h_per_cycle[cid])):.4f}")

    return {
        'h_per_cycle':         {cid: h_per_cycle[cid].tolist() for cid in cycle_ids},
        'K_gamma':             {cid: K_gamma[cid] for cid in cycle_ids},
        'k0_gamma':            k0_gamma,
        'k1_gamma':            k1_gamma,
        'K_intersection':      K_intersection,
        'k_star':              k_star,
        'delta_III':           best_delta,
        'global_min_delta_III': max_diff,
        'criterion_satisfied': True,
    }

########################## pick top-N cycles from stage-1 ##########################

def select_cycle_sets(stage1_result, n_cycles=3):
    cycle_sets = stage1_result.get('cycle_sets')
    if cycle_sets:
        return cycle_sets[:n_cycles]

    selected = stage1_result.get('selected_cycles_greedy')
    if selected:
        sets = []
        for m in range(1, min(n_cycles, len(selected)) + 1):
            sets.append(selected[:m])
        return sets

    g_gamma     = stage1_result.get('g_gamma', {})
    good_cycles = stage1_result.get('good_cycles', [])
    best_box    = stage1_result.get('best_box_cycle')

    if not good_cycles:
        return []

    ranked = sorted(good_cycles,
                    key=lambda c: g_gamma.get(str(c), float('inf')))

    if best_box is not None and best_box in ranked:
        ranked.remove(best_box)
        ranked.insert(0, best_box)

    if not ranked:
        return []

    sets = []
    for m in range(1, min(n_cycles, len(ranked)) + 1):
        sets.append(ranked[:m])
    return sets



def print_period_table(periods_all):
    counts = Counter(periods_all)
    sorted_periods = sorted(counts.items())
    print("\n" + "=" * 38)
    print(f"  {'Period':>10}  |  {'# cycles':>10}")
    print("-" * 38)
    for period, count in sorted_periods:
        print(f"  {period:>10}  |  {count:>10}")
    print("=" * 38)
    print(f"  Total cycles: {len(periods_all)}")
    print("=" * 38 + "\n")


def compute_delta_II(measures_by_cycle_over_Uk, cycle_ids, first_singleton_index):
    k_values  = list(range(first_singleton_index))
    all_pairs = [(cycle_ids[a], cycle_ids[b])
                 for a in range(len(cycle_ids))
                 for b in range(a + 1, len(cycle_ids))]

    def get_mu(k, cid):
        if cid not in measures_by_cycle_over_Uk:
            return 0.0
        arr = measures_by_cycle_over_Uk[cid]
        if k >= len(arr):
            return 0.0
        return float(arr[k])

    max_diff       = 0.0
    worst_pair     = None
    worst_k        = None
    pair_max_diffs = {}

    for ci, cj in all_pairs:
        pair_max = 0.0
        for k in k_values:
            mu_i = get_mu(k, ci)
            mu_j = get_mu(k, cj)
            diff = abs(mu_i - mu_j)
            if diff > pair_max:
                pair_max = diff
            if diff > max_diff:
                max_diff   = diff
                worst_pair = (ci, cj)
                worst_k    = k
        pair_max_diffs[(ci, cj)] = pair_max

    delta_II = max_diff / 2.0

    print(f"\n  [Validation Criterion II] delta_II computation")
    print(f"  {'='*60}")
    print(f"  k range: 0 to {first_singleton_index - 1}")
    print(f"  max |mu_i(k) - mu_j(k)| over all pairs and k = {max_diff:.6f}")
    print(f"  => smallest delta_II = {delta_II:.6f}")
    print(f"  Worst pair: {worst_pair} at k={worst_k}")

    return delta_II, pair_max_diffs

def plot_mu_gamma_over_uk(movie_sequence, U_k_all_dims, first_singleton_index,
                          periods_all, good_cycles,
                          px, py, output_dir, group, hours, r_grids, tag=''):
    """
    mu_gamma(k) = N_gamma(k) / T_gamma over k for every cycle
    """
    k_values   = list(range(first_singleton_index + 1))
    n_cycles   = len(movie_sequence)

    
    cycle_arrays = [movie_sequence[cid] for cid in range(n_cycles)]
    measures_by_cycle = {cid: np.zeros(len(k_values)) for cid in range(n_cycles)}

    for k_idx, k in enumerate(k_values):
        # Build uk as a set of byte-strings 
        uk_arr   = U_k_all_dims[k]                          # shape (n_uk, D)
        uk_bytes = set(row.tobytes() for row in uk_arr)

        for cid in range(n_cycles):
            pts = cycle_arrays[cid]                          # shape (T, D)
            T   = periods_all[cid]
            N   = sum(1 for row in pts if row.tobytes() in uk_bytes)
            measures_by_cycle[cid][k_idx] = N / T if T > 0 else 0.0

    plt.figure(figsize=(12, 7))
    for cid in range(len(movie_sequence)):
        measures = measures_by_cycle[cid]
        if cid in good_cycles:
            plt.plot(k_values, measures, color='black', linewidth=1, zorder=15,
                     label=f'Good cycle {cid}')
        else:
            plt.plot(k_values, measures, color='black', linewidth=1.2, alpha=0.6)
    plt.xlabel('Scale k', fontsize=15)
    plt.ylabel(r'$\mu_\gamma(k)$', fontsize=15)
    plt.xlim(0, first_singleton_index)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir,
        f'diag_mu_gamma_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"  [diag] mu_gamma(k) plot saved → {fname}")

    return measures_by_cycle  


def plot_ngamma_histogram(entropies_per_cycle_all, first_singleton_index,
                           px, py, output_dir, group, hours, r_grids, tag=''):
    """
    Bar chart of N_gamma(k) over U_k for a representative single cycle
    """
    k_values = list(range(first_singleton_index + 1))

    for cid, ent_dict in entropies_per_cycle_all.items():
        sizes = [ent_dict['U_k'][k]['N_gamma'].get(cid, 0) for k in k_values]
        if max(sizes) == 0:
            continue

        plt.figure(figsize=(15, 5))
        plt.bar(k_values, sizes, color='black', edgecolor='white', linewidth=0.5)
        plt.xlabel('Scale k', fontsize=15)
        plt.ylabel('Number of points', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(-1, first_singleton_index + 0.5)
        plt.ylim(0, max(sizes) * 1.15)
        for k in range(0, len(sizes), 5):
            if sizes[k] > 0:
                plt.text(k, sizes[k] + max(sizes) * 0.001, str(sizes[k]),
                         ha='center', va='bottom', fontsize=10, rotation=0)
        plt.tight_layout()
        fname = os.path.join(output_dir,
            f'diag_ngamma_cycle{cid}_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf') 
        plt.savefig(fname, format='pdf')
        plt.close()

    print(f"  [diag] N_gamma bar charts saved → {output_dir}")

def plot_uk_size_histogram(U_k_all_dims, first_singleton_index,
                           output_dir, group, hours, r_grids, tag=''):
    """
    Bar chart of |U_k| (number of points) for k = 0 … first_singleton_index
    """
    k_values = list(range(first_singleton_index + 1))
    sizes    = [len(U_k_all_dims[k]) for k in k_values]

    plt.figure(figsize=(15, 5))
    plt.bar(k_values, sizes, color='black', edgecolor='white', linewidth=0.5)
    plt.xlabel('Scale k', fontsize=15)
    plt.ylabel('Number of points', fontsize=15)
    plt.xlim(-1, first_singleton_index + 0.5)
    plt.ylim(0, max(sizes) * 1.1)
    for k in range(0, len(sizes), 5):
        plt.text(k, sizes[k] + max(sizes) * 0.001, str(sizes[k]),
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fname = os.path.join(output_dir,
        f'uk_sizes_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"  [plot] |U_k| histogram saved → {fname}")



def plot_entropy_all_aggregate(movie_sequence, periods_all,
                               U_k_all_dims, first_singleton_index,
                               px, py, output_dir, group, hours, r_grids, tag=''):
    """
    Joint entropy of ALL cycles together — single black line, marker='o'
    """
    k_values    = list(range(first_singleton_index + 1))
    all_cids    = list(range(len(movie_sequence)))
    periods_set = [periods_all[cid] for cid in all_cids]

    cache_path = os.path.join(
        output_dir,
        f'entropy_cache_joint_allcycles_fsi{first_singleton_index}'
        f'_gr{group}_{hours}h_r{r_grids}_{tag}.npy',
    )
    if os.path.exists(cache_path):
        print(f"  [cache] all-cycles aggregate loaded from {cache_path}")
        ent = np.load(cache_path, allow_pickle=True).item()
    else:
        ent = entropies_cycles(
            movie_sequence, periods_set, U_k_all_dims,
            all_cids, first_singleton_index,
        )
        np.save(cache_path, ent)
        print(f"  [cache] all-cycles aggregate saved to {cache_path}")

    entropy_values = [ent['U_k'][k]['entropy'] for k in k_values]

    plt.figure(figsize=(10, 8))
    plt.plot(k_values, entropy_values, marker='o', color='black', linewidth=1.0,
             label=f'All {len(all_cids)} cycles')
    plt.xlabel('Scale k', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    y_max = max(entropy_values) if entropy_values else 2.5
    plt.ylim(-0.01, max(2.5, y_max * 1.05))
    plt.xlim(-1, first_singleton_index + 1)
    plt.tight_layout()
    fname = os.path.join(output_dir,
        f'entropy_allcycles_aggregate_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"  [plot] entropy all cycles aggregate → {fname}")


def plot_entropy_m_last_separate(entropies_per_cycle_all, cycle_sets,
                                first_singleton_index,
                                px, py, output_dir, group, hours, r_grids, tag=''):
    """
    One coloured line per cycle in the largest greedy set
    """
    if not cycle_sets:
        print("  [plot] no cycle_sets — skipping M-separate plot.")
        return

    k_values = list(range(first_singleton_index + 1))
    m_last   = cycle_sets[-1]   # largest set, e.g. [7, 33, 13] for M=3

    plt.figure(figsize=(10, 8))
    for cid in m_last:
        ent_dict       = entropies_per_cycle_all[cid]
        entropy_values = [ent_dict['U_k'][k]['entropy'] for k in k_values]
        plt.plot(k_values, entropy_values, marker='o', label=f'Cycle {cid}', linewidth=1.0)
    plt.xlabel('Scale k', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    plt.legend(fontsize=10)
    plt.ylim(-0.01, 2.5)
    plt.xlim(-1, first_singleton_index + 1)
    plt.tight_layout()
    fname = os.path.join(output_dir,
        f'entropy_M{len(m_last)}_separate_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"  [plot] entropy M={len(m_last)} separate → {fname}")


def plot_entropy_mx_aggregate(movie_sequence, periods_all,
                              U_k_all_dims, cycle_sets,
                              first_singleton_index,
                              px, py, output_dir, group, hours, r_grids, tag=''):
    """ 
    One separate PDF per M: aggregate (joint) entropy of that cycle set
    """
    k_values = list(range(first_singleton_index + 1))

    for cset in cycle_sets:
        M           = len(cset)
        periods_set = [periods_all[cid] for cid in cset]
        cache_path  = os.path.join(
            output_dir,
            f'entropy_cache_joint_M{M}_cycles{cset[0]}to{cset[-1]}' +
            f'_fsi{first_singleton_index}_gr{group}_{hours}h_r{r_grids}_{tag}.npy',
        )
        if os.path.exists(cache_path):
            print(f"  [cache] joint entropy M={M} {cset} loaded from {cache_path}")
            ent = np.load(cache_path, allow_pickle=True).item()
        else:
            ent = entropies_cycles(
                movie_sequence, periods_set, U_k_all_dims,
                cset, first_singleton_index,
            )
            np.save(cache_path, ent)
            print(f"  [cache] joint entropy M={M} {cset} saved to {cache_path}")

        entropy_values = [ent['U_k'][k]['entropy'] for k in k_values]

        plt.figure(figsize=(10, 8))
        plt.plot(k_values, entropy_values, marker='o', color='black', linewidth=1.0,
                 label=f'M={M}: cycles {cset}')
        plt.xlabel('Scale k', fontsize=15)
        plt.ylabel('Entropy', fontsize=15)
        y_max = max(entropy_values) if entropy_values else 2.5
        plt.ylim(-0.01, max(2.5, y_max * 1.05))
        plt.xlim(-1, first_singleton_index + 1)
        plt.tight_layout()
        fname = os.path.join(output_dir,
            f'entropy_M{M}_aggregate_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
        plt.savefig(fname, format='pdf')
        plt.close()
        print(f"  [plot] entropy M={M} aggregate → {fname}")

 

def plot_entropy_all_separate(entropies_per_cycle_all, first_singleton_index,
                              px, py, output_dir, group, hours, r_grids, tag=''):
    """
    One coloured line per cycle — ALL cycles plotted separately, marker='o'
    """
    k_values  = list(range(first_singleton_index + 1))
    n_cycles  = len(entropies_per_cycle_all)
 
    plt.figure(figsize=(10, 8))
    for cid in range(n_cycles):
        # try loading directly from cache file first
        cache_path = os.path.join(
            output_dir,
            f'entropy_cache_cycle{cid}_fsi{first_singleton_index}'
            f'_gr{group}_{hours}h_r{r_grids}.npy',
        )
        if os.path.exists(cache_path):
            ent_dict = np.load(cache_path, allow_pickle=True).item()
        else:
            ent_dict = entropies_per_cycle_all[cid]
 
        entropy_values = [ent_dict['U_k'][k]['entropy'] for k in k_values]
        plt.plot(k_values, entropy_values, linewidth=1.2, alpha=0.6, color='black')
 
    plt.xlabel('Scale k', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    plt.ylim(-0.01, 2.5)
    plt.xlim(-1, first_singleton_index + 1)
    plt.tight_layout()
    fname = os.path.join(output_dir,
        f'entropy_allcycles_separate_idx{px}_idy{py}_gr{group}_{hours}h_r{r_grids}_{tag}.pdf')
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"  [plot] entropy all cycles separate → {fname}")

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',       type=int,   required=True)
    parser.add_argument('--hours',       type=int,   required=True)
    parser.add_argument('--k_grid',      type=int,   required=True)
    parser.add_argument('--file_f',      type=str,   required=True)
    parser.add_argument('--file_cyc',    type=str,   required=True)
    parser.add_argument('--stage1_dir',  type=str,   required=True)
    parser.add_argument('--output_dir',  type=str,   required=True)
    parser.add_argument('--pairs',       nargs='+',  required=True)
    parser.add_argument('--n_cycles',    type=int,   default=3)
    parser.add_argument('--deltas',      nargs='+',  type=float,
                        default=[0.10, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument('--r_grids',      type=int,   default=1500)
    parser.add_argument('--all_cycles_good', type=str, default='false', help='"true" or "false"')
    args = parser.parse_args()
    tag = 'allcyclesaregood' if args.all_cycles_good.lower() == 'true' else ''
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    data_list = np.load(args.file_f, allow_pickle=True).item()
    # print(f"The number of points in the entire list is {len(data_list['all_points'])}")

    if args.hours == 6 or args.hours == 12:
        data_list = {k: v[:720*args.hours*args.r_grids] if isinstance(v, list) else v
                    for k, v in data_list.items()}
    elif args.hours == 24:
        data_list = {k: v[:720*args.hours*args.r_grids] if isinstance(v, list) else v
                    for k, v in data_list.items()}
    print(f"The number of points after slicing is {len(data_list['all_points'])}")

    cycles         = np.load(args.file_cyc, allow_pickle=True)
    movie_sequence = get_cycle_points(cycles, data_list)
    # print(f"Loaded {len(movie_sequence)} cycles.")

    periods_all = [len(movie_sequence[cid]) for cid in range(len(movie_sequence))]

    # period table 
    print_period_table(periods_all)
    # print(f"Periods: {periods_all}")

    # U_k (computed once, shared across all pairs) 
    print("Computing U_k …")
    points = np.vstack(movie_sequence)
    U_k_all_dims = compute_uk(points, args.k_grid)
    first_singleton_index = singleton(U_k_all_dims)
    print(f"First singleton index: {first_singleton_index}")

    # per-cycle entropies for ALL cycles
    print("\nComputing per-cycle entropies for all cycles …")
    entropies_per_cycle_all = {}
    for cid in range(len(movie_sequence)):
        cache_path = os.path.join(
            args.output_dir,
            f'entropy_cache_cycle{cid}_fsi{first_singleton_index}'
            f'_gr{args.group}_{args.hours}h_r{args.r_grids}_{tag}.npy',
        )
        if os.path.exists(cache_path):
            print(f"  cycle {cid}: loading from cache → {cache_path}")
            entropies_per_cycle_all[cid] = np.load(cache_path, allow_pickle=True).item()
        else:
            print(f"  cycle {cid}: computing …")
            ent = entropies_cycles(
                movie_sequence,
                periods=[periods_all[cid]],
                U_k_alldims=U_k_all_dims,
                cycle_ids=[cid],
                first_singleton_index=first_singleton_index,
            )
            np.save(cache_path, ent)
            print(f"  cycle {cid}: saved cache → {cache_path}")
            entropies_per_cycle_all[cid] = ent

    if args.all_cycles_good.lower() == 'true':
        # Get cycle_sets from first pair (all pairs identical in true mode)
        first_pair   = args.pairs[0]
        px0, py0     = [int(x) for x in first_pair.split(',')]
        stage1_path0 = os.path.join(args.stage1_dir,
            f'bestcycles_idx{px0}_idy{py0}_r{args.r_grids}_{tag}.json')
        with open(stage1_path0) as f:
            stage1_0 = json.load(f)
        good_cycles_0 = [int(c) for c in stage1_0.get('good_cycles', [])]
        cycle_sets_0  = select_cycle_sets(stage1_0, n_cycles=args.n_cycles)

        # Diagnostics and plots — once
        print("\n--- Global diagnostics (pair-independent) ---")
        plot_uk_size_histogram(U_k_all_dims, first_singleton_index,
            args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        measures_by_cycle_over_Uk = plot_mu_gamma_over_uk(
            movie_sequence, U_k_all_dims, first_singleton_index,
            periods_all, good_cycles_0,
            px0, py0, args.output_dir, args.group, args.hours, args.r_grids, tag)

        all_cids = list(range(len(movie_sequence)))
        delta_II, _ = compute_delta_II(measures_by_cycle_over_Uk, all_cids,
                                    first_singleton_index)
        print(f"  delta_II = {delta_II:.6f}")

        plot_ngamma_histogram(entropies_per_cycle_all, first_singleton_index,
            px0, py0, args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        plot_entropy_all_separate(entropies_per_cycle_all, first_singleton_index,
            px0, py0, args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        plot_entropy_all_aggregate(movie_sequence, periods_all, U_k_all_dims,
            first_singleton_index, px0, py0,
            args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        plot_entropy_m_last_separate(entropies_per_cycle_all, cycle_sets_0,
            first_singleton_index, px0, py0,
            args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        plot_entropy_mx_aggregate(movie_sequence, periods_all, U_k_all_dims,
            cycle_sets_0, first_singleton_index, px0, py0,
            args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

        # Plateau — once per cycle set
        print("\n--- Plateau computation (pair-independent) ---")
        plateau_cache = {}
        for cycle_ids in cycle_sets_0:
            M   = len(cycle_ids)
            key = tuple(cycle_ids)
            print(f"\n  M={M}: cycle_ids={cycle_ids}")
            entropies_per_cycle = {cid: entropies_per_cycle_all[cid]
                                for cid in cycle_ids}
            plateau = compute_plateau(entropies_per_cycle, cycle_ids,
                                    first_singleton_index)
            if not plateau['criterion_satisfied']:
                print(f"  ⚠ No valid delta_III for M={M}. Skipping.")
            plateau_cache[key] = plateau

        # Save plateau JSON per pair (filename includes px,py for stage 3 lookup)
        for pair_str in args.pairs:
            px, py = [int(x) for x in pair_str.split(',')]
            print(f"\n{'='*60}\nPair ({px}, {py}) — saving plateau JSONs")

            stage1_path = os.path.join(args.stage1_dir,
                f'bestcycles_idx{px}_idy{py}_r{args.r_grids}_{tag}.json')
            if not os.path.exists(stage1_path):
                print(f"  Stage-1 file not found. Skipping.")
                continue

            with open(stage1_path) as f:
                stage1 = json.load(f)
            cycle_sets = select_cycle_sets(stage1, n_cycles=args.n_cycles)

            for cycle_ids in cycle_sets:
                M       = len(cycle_ids)
                key     = tuple(cycle_ids)
                plateau = plateau_cache.get(key)

                if plateau is None or not plateau.get('criterion_satisfied'):
                    print(f"  M={M}: no valid plateau. Skipping.")
                    continue

                result = {
                    'projection_x':          px,
                    'projection_y':          py,
                    'cycle_ids':             cycle_ids,
                    'delta_III':             plateau['delta_III'],
                    'criterion_satisfied':   True,
                    'k_star':                plateau['k_star'],
                    'K_intersection':        plateau['K_intersection'],
                    'K_gamma':               {str(cid): plateau['K_gamma'][cid]
                                            for cid in cycle_ids},
                    'first_singleton_index': first_singleton_index,
                }
                out_path = os.path.join(args.output_dir,
                    f'plateaux_gr{args.group}_idx{px}_idy{py}_M{M}'
                    f'_r{args.r_grids}_{tag}.json')
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved → {out_path}")

    else:
        for pair_str in args.pairs:
            px, py = [int(x) for x in pair_str.split(',')]
            print(f"\n{'='*60}\nPair ({px}, {py})")

            stage1_path = os.path.join(args.stage1_dir,
                f'bestcycles_idx{px}_idy{py}_r{args.r_grids}_{tag}.json')
            if not os.path.exists(stage1_path):
                print(f"  Stage-1 file not found. Skipping.")
                continue

            with open(stage1_path) as f:
                stage1 = json.load(f)
            good_cycles = [int(c) for c in stage1.get('good_cycles', [])]
            cycle_sets  = select_cycle_sets(stage1, n_cycles=args.n_cycles)

            if not cycle_sets:
                print("  No good cycles. Skipping.")
                continue

            print(f"  good_cycles: {good_cycles}")
            print(f"  cycle_sets:  {cycle_sets}")

            measures_by_cycle_over_Uk = plot_mu_gamma_over_uk(
                movie_sequence, U_k_all_dims, first_singleton_index,
                periods_all, good_cycles,
                px, py, args.output_dir, args.group, args.hours, args.r_grids, tag)

            all_cids = list(range(len(movie_sequence)))
            delta_II, _ = compute_delta_II(measures_by_cycle_over_Uk, all_cids,
                                        first_singleton_index)
            print(f"  delta_II = {delta_II:.6f}")

            plot_ngamma_histogram(entropies_per_cycle_all, first_singleton_index,
                px, py, args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

            plot_entropy_m_last_separate(entropies_per_cycle_all, cycle_sets,
                first_singleton_index, px, py,
                args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

            plot_entropy_all_separate(entropies_per_cycle_all, first_singleton_index,
                px, py, args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

            plot_entropy_all_aggregate(movie_sequence, periods_all, U_k_all_dims,
                first_singleton_index, px, py,
                args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

            plot_entropy_mx_aggregate(movie_sequence, periods_all, U_k_all_dims,
                cycle_sets, first_singleton_index, px, py,
                args.output_dir, args.group, args.hours, args.r_grids, tag=tag)

            print(f"\n  --- Plateau computation ---")
            for cycle_ids in cycle_sets:
                M = len(cycle_ids)
                print(f"\n  M={M}: cycle_ids={cycle_ids}")
                entropies_per_cycle = {cid: entropies_per_cycle_all[cid]
                                    for cid in cycle_ids}
                plateau = compute_plateau(entropies_per_cycle, cycle_ids,
                                        first_singleton_index)
                if not plateau['criterion_satisfied']:
                    print(f"  ⚠ No valid delta_III for M={M}. Skipping.")
                    continue

                result = {
                    'projection_x':          px,
                    'projection_y':          py,
                    'cycle_ids':             cycle_ids,
                    'delta_III':             plateau['delta_III'],
                    'criterion_satisfied':   True,
                    'k_star':                plateau['k_star'],
                    'K_intersection':        plateau['K_intersection'],
                    'K_gamma':               {str(cid): plateau['K_gamma'][cid]
                                            for cid in cycle_ids},
                    'first_singleton_index': first_singleton_index,
                }
                out_path = os.path.join(args.output_dir,
                    f'plateaux_gr{args.group}_idx{px}_idy{py}_M{M}'
                    f'_r{args.r_grids}_{tag}.json')
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved → {out_path}")

if __name__ == '__main__':
    main()