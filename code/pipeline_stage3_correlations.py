"""
Cross-correlation analysis, as described in Section 3.2 of the paper.

For every (pair, M) combination it builds the transition matrix P_k
and stationary measure mu at each k* in plateau, then computes the
delayed correlation Corr^(s)(xi, xi_tilde) up to --abs_max_delay steps.


Usage:
  python pipeline_stage3_correlations.py \\
      --group 0 --hours 6 --k_grid 5 \\
      --file_f   <allpoints.npy> \\
      --file_cyc <cycles.npy> \\
      --stage1_dir <stage1_dir> \\
      --stage2_dir <stage2_dir> \\
      --output_dir <output_dir> \\
      --pairs "113,97" "97,113" "152,153" \\
      [--M_values 2 3] [--abs_max_delay 50000] \\
      [--zero_tol 1e-10] [--zero_run 100]
"""

import argparse
import glob
import json
import os
import sys
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs


import multiprocessing as mp


PALETTE = [
    '#1f77b4', '#2ca02c',  '#ff7f0e', '#d62728',  '#9467bd', '#8c564b', '#e377c2', '#17becf',  
]


def get_cycle_points(cycles, data_f):
    cycle_points = pd.DataFrame(columns=['movie_points'])
    for i in range(len(cycles)):
        unique_cycle_indices = cycles[i]
        if len(unique_cycle_indices) == 1:
            print(f"Skipping cycle {i} with only one point.")
            continue
        cycle_points.loc[i] = [ [data_f['all_points'][idx] for idx in unique_cycle_indices] ]
        cycle_points.at[i, 'movie_points'] = np.stack([data_f['all_points'][idx] for idx in unique_cycle_indices])

    return cycle_points


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


########################## transition matrix and correlations ##########################

def build_graph_and_correlations_newPk(U_k, cycle_points_test, cycle_ids, periods,
                                  coord_idx, coord_idy, k_star, max_delay=2000, r=1000, tag=""):
    
    period_map = {cid: periods[i] for i, cid in enumerate(cycle_ids)}
    num_cycles = len(cycle_ids)
    uk_set     = {tuple(U_k[i]) for i in range(len(U_k))}
    num_points_uk_cycles = 0

    # 1: Find visit times and N_gamma
    visit_times_map = {}
    N_gamma_map     = {}

    for cid in cycle_ids:
        cycle_pts = cycle_points_test['movie_points'].iloc[cid]
        T         = len(cycle_pts)
        visits    = [t for t in range(T) if tuple(cycle_pts[t]) in uk_set]
        visit_times_map[cid] = visits
        N_gamma_map[cid]     = len(visits)

        cycle_set = {tuple(row) for row in cycle_pts}
        matching_points = cycle_set & uk_set
        num_points_uk_cycles += N_gamma_map[cid]


    # 2: Build excursions using explicit per-excursion vertices
    excursions_points = {cid: [] for cid in cycle_ids}  # cid -> list of excursions (list of point tuples)
    E_gamma_map = {}

    for cid in cycle_ids:
        cycle_pts = cycle_points_test['movie_points'].iloc[cid]
        T = len(cycle_pts)
        visits = visit_times_map[cid]
        n_visits = len(visits)
        assigned_t = set()

        for i in range(n_visits):
            t_start = visits[i]
            t_end = visits[(i + 1) % n_visits]

            between_points = []
            t = (t_start + 1) % T
            steps = 0
            while t != t_end and steps < T:
                if t not in assigned_t:
                    key = tuple(cycle_pts[t])
                    if key not in uk_set:
                        between_points.append(key)
                        assigned_t.add(t)

                t = (t + 1) % T
                steps += 1

            excursions_points[cid].append(between_points)

        E_gamma_map[cid] = sum(1 for exc in excursions_points[cid] if len(exc) > 0)

        total_excursion_pts = sum(len(exc) for exc in excursions_points[cid])
        expected = period_map[cid] - N_gamma_map[cid]
        if total_excursion_pts != expected:
            print(f"WARNING cycle {cid}: excursion points={total_excursion_pts} "
                  f"but expected T-N_gamma={expected}. "
                  f"Diff={expected - total_excursion_pts} points missing.")
            all_excursion_t = assigned_t
            uk_t = set(visit_times_map[cid])
            all_t = set(range(T))
            missed_t = all_t - all_excursion_t - uk_t
            if missed_t:
                print(f"Missed time steps: {sorted(missed_t)[:10]}...")


    total_periods = sum(period_map[cid] for cid in cycle_ids)
    shape_P = total_periods - num_points_uk_cycles + 1

    vertex_cycle = {}   # graph index -> cid
    index_to_point = {} # graph index -> point tuple
    excursions_indices = {cid: [] for cid in cycle_ids}

    current_idx = 1
    for cid in cycle_ids:
        for excursion_pts in excursions_points[cid]:
            idx_list = []
            for pt in excursion_pts:
                vertex_cycle[current_idx] = cid
                index_to_point[current_idx] = pt
                idx_list.append(current_idx)
                current_idx += 1
            excursions_indices[cid].append(idx_list)

    N_graph = current_idx
    N_nonroot = N_graph - 1
    total_periods = sum(period_map[cid] for cid in cycle_ids)

    # 3: Build transition matrix P_k
    P_k = lil_matrix((shape_P, shape_P), dtype=float)
    denom = sum(N_gamma_map[cid] / period_map[cid] for cid in cycle_ids)

    if denom == 0.0:
        raise ValueError("denom = Σ N_γ/T_γ is zero — no cycle visits U_k")

    row0_offdiag_sum = 0.0

    for cid in cycle_ids:
        T = period_map[cid]
        w = 1.0 / (T * denom)   # same weight for every excursion start in this cycle

        for excursion in excursions_indices[cid]:
            if len(excursion) == 0:
                continue

            # Root → first vertex of this excursion
            P_k[0, excursion[0]] += w
            row0_offdiag_sum     += w

            # Deterministic chain: each internal vertex goes to next
            for j in range(len(excursion) - 1):
                P_k[excursion[j], excursion[j+1]] = 1.0

            # Last vertex of excursion → root
            P_k[excursion[-1], 0] = 1.0

    # Self-loop on root
    P_k[0, 0] = 1.0 - row0_offdiag_sum
    print(f"P_k[0,0] = {P_k[0,0]:.6f}, row0_offdiag_sum = {row0_offdiag_sum:.6f}")
    print(f"P_k[0, 0] {P_k[0,0]:.6f} should be also same as 1- sum_cycles escape points/ num_points_uk_cycles = {1 - sum(E_gamma_map.values()) / sum(N_gamma_map.values())}")  # sum(period_map.values()):.6f}")
    print(f"Total row 0 sum: {P_k[0, :].sum():.6f} (should be 1.0)")

    # Verify rows sum to 1
    P_k_csr = P_k.tocsr()
    row_sums = np.array(P_k_csr.sum(axis=1)).flatten()
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-8)[0]
    if len(bad_rows) > 0:
        print(f"WARNING: {len(bad_rows)} rows do not sum to 1:")
        for r in bad_rows[:5]:
            print(f"  row {r}: sum={row_sums[r]:.6f}, "
                  f"cycle={vertex_cycle.get(r, 'root')}")
    else:
        print(f"P_k rows all sum to 1 ✓")

    if N_graph != shape_P:
        print(f"WARNING: shape mismatch. expected shape_P={shape_P}, built N_graph={N_graph}")

    top3 = []


    # 4: Build measure mu
    mu    = np.zeros(shape_P, dtype=float)
    mu[0] = sum(N_gamma_map[cid] / period_map[cid]
                for cid in cycle_ids) / num_cycles

    for v_idx, cid in vertex_cycle.items():
        mu[v_idx] = 1.0 / (period_map[cid] * num_cycles)

    mu_sum = mu.sum()
    if abs(mu_sum - 1.0) > 1e-8:
        print(f"WARNING: mu sums to {mu_sum:.6f}, not 1.")
    else:
        print(f"mu sums to 1 ✓")

    mu_after = P_k_csr.T.dot(mu)  # sparse equivalent of mu @ P_k
    print(f"mu @ P_k should match mu so the norm of difference is {np.linalg.norm(mu_after - mu)}")
    if not np.allclose(mu, mu_after, atol=1e-6):
        print(f"WARNING: mu not left eigenvalue of P_k. "
              f"mu @ P_k = {mu_after[:5]}... (should match mu[:5]={mu[:5]}...)")


    # 5: Build xi over graph vertices
    # Non-root: xi(v) = v[coord]
    # Root:     xi(v_0) = 1/|Gamma*| * sum_gamma 1/T_gamma * sum_t xi(cycle_pts[t])
    def build_xi(coord):
        xi = np.zeros(shape_P, dtype=float)

        # Non-root: use coord-th element of the point tuple
        for v_idx, pt_tuple in index_to_point.items():
            if coord >= len(pt_tuple):
                raise IndexError(f"coord={coord} out of range for pt_tuple len={len(pt_tuple)}, v_idx={v_idx}, pt_tuple[:4]={pt_tuple[:4]}")
            xi[v_idx] = pt_tuple[coord]


        # Root: weighted average over all cycle points interesected to uk
        root_num = 0.0
        root_denom = 0.0
        for cid in cycle_ids:
            cycle_pts = cycle_points_test['movie_points'].iloc[cid]
            T         = period_map[cid]
            visits = visit_times_map[cid]
            for t in visits:
                root_num += cycle_pts[t][coord] / T
            root_denom += N_gamma_map[cid] / T
        xi[0] = root_num / root_denom if root_denom != 0.0 else 0.0

        return xi

    xi       = build_xi(coord_idx)
    xi_tilde = build_xi(coord_idy)

    # 6: Center xi under mu
    mean_xi = np.dot(xi, mu)
    xi      = xi - mean_xi
    xi_mu   = xi * mu
    vec     = xi_mu.copy()
    print(f"xi centered: mean = {np.dot(xi, mu):.2e} (should be ~0)")

    var_xi = np.dot(xi**2, mu)

    if var_xi == 0.0:
        print("Zero var_xi. Returning zeros.")
        return np.arange(max_delay + 1), np.zeros(max_delay + 1)

    # STEP 7: Compute delayed correlations
    vec           = xi_mu.copy()          # vec_s = P_k^s @ (xi*mu)

    xi_tilde_centered_0 = xi_tilde - np.dot(xi_tilde, mu)
    xi_tilde_s    = xi_tilde_centered_0.copy()   # will be advanced each step
    print(f"Centered xi tilde initial: mean={np.dot(xi_tilde_s, mu):.2e} (should be ~0)")
    var_xi_tilde = np.dot(xi_tilde_centered_0**2, mu)  # fixed, computed once at s=0
    denominator  = np.sqrt(var_xi) * np.sqrt(var_xi_tilde)  # constant for all s


    correlations  = np.zeros(max_delay + 1)

    for s in range(max_delay + 1):
        numerator = np.dot(xi_mu, xi_tilde_s)
        correlations[s] = numerator / denominator
        xi_tilde_s = P_k_csr.dot(xi_tilde_s)  # advance xi_tilde_s by one step of P_k for next iteration

        if s % 100 == 0:
            print(f"  s={s:4d}: Corr^(s)={correlations[s]:.10f}, "
                  f"var_xi_tilde={var_xi_tilde:.10f}")

        # Advance xi_tilde by one step of P_k
        xi_tilde_s   = P_k_csr.dot(xi_tilde_s)

    zero_indices = np.where(correlations < 1e-10)[0]
    if len(zero_indices) > 0:
        print(f"First zero correlation (< 1e-10) encountered at delay s={zero_indices[0]}")
    else:
        print(f"No correlation dropped below 1e-10 within the computed delay range "
            f"(max delay={len(correlations)-1}, min |corr|={np.min(np.abs(correlations)):.2e})")
    return np.arange(max_delay + 1), correlations, top3



def _load_corr_files(output_dir, px, py, M, k_star_list, max_delay_computed, grids, group, hours, tag):
    if k_star_list is not None:
        candidates = []
        for k_star in k_star_list:
            fname = os.path.join(
                output_dir,
                f"corr_idx{px}_idy{py}_M{M}_k{k_star}"
                f"_gr{group}_{hours}h_grids{grids}_{tag}.npy",
            )
            # Try exact name first (group/r_grids baked in at save time)
            # Fall back to a glob so the helper works even if called from
            # outside main() where group/r_grids are known.
            exact_candidates = glob.glob(fname)
            if not exact_candidates:
                print(f"  Warning: no file found for k*={k_star} "
                      f"(pattern: {fname})")
            candidates.extend((k_star, f) for f in exact_candidates)
    else:
        pattern = os.path.join(
            output_dir,
            f"corr_idx{px}_idy{py}_M{M}_k*_gr{group}_{hours}h_grids{grids}_{tag}.npy",
        )
        found = sorted(glob.glob(pattern))
        candidates = []
        for fname in found:
            base  = os.path.basename(fname)
            k_str = base.split('_k')[1].split('_')[0]
            candidates.append((int(k_str), fname))

    candidates = sorted(candidates, key=lambda x: x[0])

    result = []
    for k_star, fname in candidates:
        raw  = np.load(fname, allow_pickle=True)
        data = raw.item()
        delays = np.array(data['delays'])
        corrs  = np.array(data['correlations'])
        print(f"  Loaded k*={k_star}: {len(delays)} delay values, "
              f"corr range [{corrs.min():.4f}, {corrs.max():.4f}]")
        result.append((k_star, delays, corrs))

    return result


def plot_correlations_from_files(output_dir, hours, px, py,
                                  cycle_ids, group, r_grids,
                                  k_star_list=None, max_delay=None, tag=None):
    """
    Overlaid plot: all k* curves on a single axes
    """
    M = len(cycle_ids)
    entries = _load_corr_files(output_dir, px, py, M, k_star_list, max_delay, r_grids, group, hours, tag)
    if not entries:
        print(f"  No correlation files found for idx={px}, idy={py}, M={M}. "
              f"Skipping overlaid plot.")
        return

    if len(entries) <= len(PALETTE):
        colors = PALETTE[:len(entries)]
    else:
        colors = [cm.viridis(i / (len(entries) - 1)) for i in range(len(entries))]
    fig, ax = plt.subplots(figsize=(10, 5))

    last_delay = 0
    for (k_star, delays, corrs), color in zip(entries, colors):
        d = delays[:max_delay] if max_delay is not None else delays
        c = corrs[:max_delay]  if max_delay is not None else corrs
        ax.plot(d, c, linewidth=1, color=color, label=f'k*={k_star}')
        last_delay = max(last_delay, d[-1])

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('Delay s', fontsize=15)
    ax.set_ylabel(r'$\mathrm{Corr}^{(s)}(\xi, \tilde{\xi})$', fontsize=15)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-5, last_delay + 5)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=10, ncol=3,
          bbox_to_anchor=(1.01, 1), loc='upper left',
          framealpha=0.7, handlelength=1.0)
    plt.tight_layout()

    if k_star_list:
        k_tag = f"{min(k_star_list)}-{max(k_star_list)}"
    else:
        k_tag = "kall"
    savename = os.path.join(
        output_dir,
        f"plot_corr_overlaid_idx{px}_idy{py}_M{M}"
        f"_kstar{k_tag}_maxdelay{max_delay}_gr{group}_grids{r_grids}_{tag}.pdf",
    )
    plt.savefig(savename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Overlaid plot saved → {savename}")

    k_mid = k_star_list[len(k_star_list) // 2]
    print(f"  Middle k* for separate plot: {k_mid}")
    entry_mid = [(k, d, c) for (k, d, c) in entries if k == k_mid]
    if entry_mid:
        _, delays_m, corrs_m = entry_mid[0]
        d_plot = delays_m[:max_delay] if max_delay else delays_m
        c_plot = corrs_m[:max_delay] if max_delay else corrs_m
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(d_plot, c_plot, linewidth=1.2, color='steelblue')
        ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
        ax2.set_xlabel('Delay s', fontsize=15)
        ax2.set_ylabel(r'$\mathrm{Corr}^{(s)}(\xi, \tilde{\xi})$', fontsize=15)
        ax2.set_ylim(-1, 1)
        ax2.set_xlim(-2, len(d_plot) + 2)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.legend([f'k*={k_mid} (middle)'], fontsize=10,
                loc='upper right', framealpha=0.7)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        savename2 = os.path.join(
            output_dir,
            f'plot_corr_middlek_idx{px}_idy{py}_M{M}'
            f'_k{k_mid}_maxdelay{max_delay}_gr{group}_grids{r_grids}_{tag}.pdf'
        )
        plt.savefig(savename2, format='pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'  Middle-k plot saved -> {savename2}')


def plot_correlations_separate(output_dir, hours, px, py,
                                cycle_ids, group, r_grids,
                                k_star_list=None, max_delay=None, tag=None):
    """
    Separate-subplot plot: one subplot per k*
    """
    M = len(cycle_ids)
    entries = _load_corr_files(output_dir, px, py, M, k_star_list, max_delay, r_grids, group, hours, tag)
    if not entries:
        print(f"  No correlation files found for idx={px}, idy={py}, M={M}. "
              f"Skipping separate plot.")
        return

    n_files = len(entries)
    n_cols  = min(3, n_files)
    n_rows  = int(np.ceil(n_files / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=False, sharey=False,
    )
    axes = np.array(axes).flatten()

    for i, (k_star, delays, corrs) in enumerate(entries):
        d = delays[:max_delay] if max_delay is not None else delays
        c = corrs[:max_delay]  if max_delay is not None else corrs

        ax = axes[i]
        ax.plot(d, c, linewidth=0.8, color='black')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-8, d[-1] + 8)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(f'k* = {k_star}', fontsize=18)
        ax.set_xlabel('Delay s', fontsize=15)
        ax.set_ylabel(r'$\mathrm{Corr}^{(s)}(\xi, \tilde{\xi})$', fontsize=15)

    for j in range(n_files, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if k_star_list:
        k_tag = f"{min(k_star_list)}-{max(k_star_list)}"
    else:
        k_tag = "kall"

    savename = os.path.join(
        output_dir,
        f"plot_corr_separate_idx{px}_idy{py}_M{M}"
        f"_kstar{k_tag}_maxdelay{max_delay}_gr{group}_grids{r_grids}_{tag}.pdf",
    )
    plt.savefig(savename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Separate plot saved → {savename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',          type=int,   required=True)
    parser.add_argument('--hours',          type=int,   required=True)
    parser.add_argument('--k_grid',         type=int,   required=True)
    parser.add_argument('--file_f',         type=str,   required=True)
    parser.add_argument('--file_cyc',       type=str,   required=True)
    parser.add_argument('--stage1_dir',     type=str,   required=True)
    parser.add_argument('--stage2_dir',     type=str,   required=True)
    parser.add_argument('--output_dir',     type=str,   required=True)
    parser.add_argument('--pairs',          nargs='+',  required=True)
    parser.add_argument('--M_values',       nargs='+',  type=int, default=[2, 3])
    parser.add_argument('--abs_max_delay',  type=int,   default=50000)
    parser.add_argument('--zero_tol',       type=float, default=1e-10)
    parser.add_argument('--k_star_override', type=int, default=None,
                        help='If set, use only this k* instead of all K_intersection values')
    parser.add_argument('--zero_run',       type=int,   default=100)
    parser.add_argument('--r_grids',        type=int,   default=1000)
    parser.add_argument('--all_cycles_good', type=str, default='false',
                    help='"true" or "false"')
    args = parser.parse_args()
    tag = 'allcyclesaregood' if args.all_cycles_good.lower() == 'true' else ''

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data …")
    data_list = np.load(args.file_f,   allow_pickle=True).item()
    print(f"The number of points in the entire list is {len(data_list['all_points'])}")
    R = args.r_grids

    #################### check here!!!!!!!!!!!!!!!!!
    data_list = {k: v[:720*6*R] if isinstance(v, list) else v for k, v in data_list.items()}
    print(f"The number of points after slicing is {len(data_list['all_points'])}")

    cycles    = np.load(args.file_cyc, allow_pickle=True)
    movie_sequence = get_cycle_points(cycles, data_list)
    # print(f"Loaded {len(movie_sequence)} cycles.")
    periods_all = [len(movie_sequence['movie_points'].values[i]) for i in range(len(movie_sequence))]
    # print(f"Cycle periods: {periods_all}")
    periods_all_cycles = [len(cycles[cid]) for cid in range(len(cycles))]
    # print(f"Cycle periods (all) based on cycles: {periods_all_cycles}")

    print("Computing U_k …")
    points = np.vstack(movie_sequence['movie_points'].values).astype(float)

    U_k_all_dims = compute_uk(points, args.k_grid)
    first_singleton_index = singleton(U_k_all_dims)
    print(f"First singleton index: {first_singleton_index}")

    summary = []  # collect (pair, M, cycle_ids, k_star, effective_max_delay)

    for pair_str in args.pairs:
        px_str, py_str = pair_str.split(',')
        px, py = int(px_str), int(py_str)
        print(f"\n{'='*60}")
        print(f"Pair ({px}, {py})")

        stage1_path = os.path.join(args.stage1_dir,
                                   f'bestcycles_idx{px}_idy{py}_r{args.r_grids}_{tag}.json')
        if not os.path.exists(stage1_path):
            print(f"  Stage-1 file missing: {stage1_path}. Skipping.")
            continue
        print(f"Pair ({px}, {py})")

        pair_k_stars_by_M = {}

        for M in args.M_values:
            stage2_path = os.path.join(args.stage2_dir,
                                       f'plateaux_gr{args.group}_idx{px}_idy{py}_M{M}_r{args.r_grids}_{tag}.json')
            if not os.path.exists(stage2_path):
                print(f"  Stage-2 file missing for M={M}: {stage2_path}. Skipping.")
                continue

            with open(stage2_path) as f:
                stage2 = json.load(f)

            cycle_ids      = stage2['cycle_ids']
            K_intersection = stage2['K_intersection']
            k_star         = stage2['k_star']

            if not K_intersection:
                print(f"  M={M}: empty K_intersection. Skipping.")
                continue

            print(f"\n  M={M}: cycle_ids={cycle_ids}, k*={k_star}, "
                  f"K_intersection={K_intersection}")

            periods = [periods_all[cid] for cid in cycle_ids]
            M_val   = len(cycle_ids)
            pair_k_stars_by_M.setdefault(M_val, [])

            k_star_list = [args.k_star_override] if args.k_star_override is not None else K_intersection
            for k_star_val in k_star_list:
                print(f"\n  --- k*={k_star_val}")
                U_k = U_k_all_dims[k_star_val]
                print(f"  |U_k|={len(U_k)}")

                # check if already computed
                corr_fname = (f'corr_idx{px}_idy{py}_M{M_val}_k{k_star_val}'
                            f'_gr{args.group}_{args.hours}h'
                            f'_grids{args.r_grids}_{tag}.npy')
                corr_fpath = os.path.join(args.output_dir, corr_fname)
                if os.path.exists(corr_fpath):
                    print(f'  Skipping ({px},{py}) M={M_val} k*={k_star_val} — already saved.')
                    saved = np.load(corr_fpath, allow_pickle=True).item()
                    delays, corrs = saved['delays'], saved['correlations']
                    top3 = saved.get('top3_eigenvalues', None)
                else:
                    delays, corrs, top3 = build_graph_and_correlations_newPk(
                            U_k=U_k,
                            cycle_points_test=movie_sequence,
                            cycle_ids=cycle_ids,
                            periods=periods,
                            coord_idx=px,
                            coord_idy=py,
                            k_star=k_star_val,
                            max_delay=args.abs_max_delay,
                            r=args.r_grids,
                            tag=tag
                        )

                    npy_fname = (f"corr_idx{px}_idy{py}_M{M_val}_k{k_star_val}"
                        f"_gr{args.group}_{args.hours}h_grids{args.r_grids}_{tag}.npy")
                    npy_path = os.path.join(args.output_dir, npy_fname)
                    tmp_npy = npy_path.replace('.npy', '_tmp.npy')
                    np.save(tmp_npy, {'delays': delays, 'correlations': corrs,
                                        'effective_max_delay': delays[-1],
                                        'cycle_ids': cycle_ids,
                                        'k_star': k_star_val,
                                        'top3_eigenvalues': top3 if 'top3' in dir() else None})
                    os.replace(tmp_npy, npy_path)
                    print(f"  Data saved → {npy_path}")

            pair_k_stars_by_M[M_val].append(k_star_val)

            summary.append({
                'pair':                f"{px},{py}",
                'M':                   M_val,
                'cycle_ids':           cycle_ids,
                'k_star':              k_star_val,
                'effective_max_delay': int(delays[-1]),
            })


            k_star_list_for_M = pair_k_stars_by_M.get(M_val, []) or None
            print(f"\n  Generating overlaid plot for M={M_val} …")
            plot_correlations_from_files(
                output_dir=args.output_dir,
                hours=args.hours,
                px=px, py=py,
                cycle_ids=cycle_ids,
                group=args.group,
                r_grids=args.r_grids,
                k_star_list=K_intersection, #k_star_list_for_M,
                max_delay=args.abs_max_delay,
                tag=tag
            )


    summary_path = os.path.join(args.output_dir, f'pipeline_summary_r{args.r_grids}_{tag}.json')
    tmp_summary = summary_path.replace('.json', '_tmp.json')
    with open(tmp_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_summary, summary_path)
    print(f"\nPipeline summary saved → {summary_path}")


if __name__ == '__main__':
    main()
