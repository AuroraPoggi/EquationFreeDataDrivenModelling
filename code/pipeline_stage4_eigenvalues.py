"""
Eigenvalue computation of transition matrices P_k for each (pair, M, k*) combination.

Usage:
  python pipeline_stage4_eigenvalues.py \\
      --group 0 --hours 6 --k_grid 5 \\
      --file_f   <allpoints.npy> \\
      --file_cyc <cycles.npy> \\
      --stage1_dir <stage1_dir> \\
      --stage2_dir <stage2_dir> \\
      --output_dir <output_dir> \\
      --pairs "35,35" "36,36" ... \\
      --M_values 30 \\
      --r_grids 1000 \\
      --all_cycles_good true \\
      --n_eigs 10
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs, ArpackNoConvergence
import shutil



def get_cycle_points(cycles, data_f):
    cycle_points = pd.DataFrame(columns=['movie_points'])
    for i in range(len(cycles)):
        unique_cycle_indices = cycles[i]
        if len(unique_cycle_indices) == 1:
            continue
        cycle_points.loc[i] = [[data_f['all_points'][idx] for idx in unique_cycle_indices]]
        cycle_points.at[i, 'movie_points'] = np.stack(
            [data_f['all_points'][idx] for idx in unique_cycle_indices]
        )
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



##########################  build P_k only ########################## 

def build_Pk(U_k, cycle_points_test, cycle_ids, periods):
    period_map = {cid: periods[i] for i, cid in enumerate(cycle_ids)}
    num_cycles = len(cycle_ids)
    uk_set     = {tuple(U_k[i]) for i in range(len(U_k))}
    num_points_uk_cycles = 0

    # 1: visit times
    visit_times_map = {}
    N_gamma_map     = {}
    for cid in cycle_ids:
        cycle_pts = cycle_points_test['movie_points'].iloc[cid]
        T         = len(cycle_pts)
        visits    = [t for t in range(T) if tuple(cycle_pts[t]) in uk_set]
        visit_times_map[cid] = visits
        N_gamma_map[cid]     = len(visits)
        num_points_uk_cycles += len(visits)

    # 2: excursions
    excursions_points = {cid: [] for cid in cycle_ids}
    E_gamma_map       = {}
    for cid in cycle_ids:
        cycle_pts  = cycle_points_test['movie_points'].iloc[cid]
        T          = len(cycle_pts)
        visits     = visit_times_map[cid]
        n_visits   = len(visits)
        assigned_t = set()
        for i in range(n_visits):
            t_start = visits[i]
            t_end   = visits[(i + 1) % n_visits]
            between = []
            t = (t_start + 1) % T
            steps = 0
            while t != t_end and steps < T:
                if t not in assigned_t:
                    key = tuple(cycle_pts[t])
                    if key not in uk_set:
                        between.append(key)
                        assigned_t.add(t)
                t = (t + 1) % T
                steps += 1
            excursions_points[cid].append(between)
        E_gamma_map[cid] = sum(1 for exc in excursions_points[cid] if exc)

    total_periods = sum(period_map[cid] for cid in cycle_ids)
    shape_P       = total_periods - num_points_uk_cycles + 1

    # 3: vertex index mapping
    excursions_indices = {cid: [] for cid in cycle_ids}
    current_idx = 1
    for cid in cycle_ids:
        for excursion_pts in excursions_points[cid]:
            idx_list = []
            for pt in excursion_pts:
                idx_list.append(current_idx)
                current_idx += 1
            excursions_indices[cid].append(idx_list)

    # 4: build P_k
    denom = sum(N_gamma_map[cid] / period_map[cid] for cid in cycle_ids)
    if denom == 0.0:
        raise ValueError("denom = Σ N_γ/T_γ is zero — no cycle visits U_k")

    P_k = lil_matrix((shape_P, shape_P), dtype=float)
    row0_offdiag_sum = 0.0
    for cid in cycle_ids:
        T = period_map[cid]
        w = 1.0 / (T * denom)
        for excursion in excursions_indices[cid]:
            if not excursion:
                continue
            P_k[0, excursion[0]] += w
            row0_offdiag_sum     += w
            for j in range(len(excursion) - 1):
                P_k[excursion[j], excursion[j + 1]] = 1.0
            P_k[excursion[-1], 0] = 1.0
    P_k[0, 0] = 1.0 - row0_offdiag_sum

    P_k_csr  = P_k.tocsr()
    row_sums = np.array(P_k_csr.sum(axis=1)).flatten()
    bad      = np.where(np.abs(row_sums - 1.0) > 1e-8)[0]
    if len(bad):
        print(f"    WARNING: {len(bad)} rows don't sum to 1")
    else:
        print(f"    P_k ({shape_P}x{shape_P}) rows sum to 1 ✓")

    return P_k_csr, shape_P, N_gamma_map, E_gamma_map



########################## eigenvalue computation ########################## 
def compute_eigenvalues(P_k_csr, shape_P, n_eigs=3):
    """
    Compute top 3 eigenvalues of P_k_csr.
    Strategy: large ncv (key for convergence on big matrices) + hard maxiter cap.
    Falls back to whatever converged if full convergence not reached.
    """
    k   = min(3, shape_P - 2)
    # ncv much larger than default — this is the main lever for convergence speed
    ncv = min(max(4 * k + 20, 200), shape_P - 1)
    v0 =  np.ones(shape_P, dtype=float) / np.sqrt(shape_P)

    # hard cap: never more than 500 iterations regardless of matrix size
    MAXITER = min(shape_P * 5, shape_P* 20)

    best = []
    for attempt, (tol, maxiter) in enumerate([
        (1e-6, 1000),   # tight: correct eigenvalues
        (1e-4, 2000),   # medium fallback
        (1e-3, 5000),   # loose fallback
    ]):
        try:
            eigenvalues, _ = eigs(P_k_csr, k=k, which='LM',
                                  tol=tol, maxiter=MAXITER, ncv=ncv, v0=v0)
            top = sorted(eigenvalues.tolist(), key=lambda x: abs(x), reverse=True)
            print(f"    Eigenvalues converged (attempt {attempt+1}, tol={tol}):")
            for i, ev in enumerate(top[:3]):
                print(f"      {i+1}: {ev.real:+.10f} + {ev.imag:+.10f}i  "
                      f"(|λ| = {abs(ev):.10f})")
            return [{'real': float(ev.real), 'imag': float(ev.imag),
                     'abs': float(abs(ev))} for ev in top]
        except ArpackNoConvergence as e:
            recovered = sorted(e.eigenvalues.tolist(),
                               key=lambda x: abs(x), reverse=True)
            print(f"    ARPACK partial (attempt {attempt+1}, tol={tol}): "
                  f"{len(recovered)} eigenvalues recovered.")
            if len(recovered) > len(best):
                best = recovered
            if len(best) >= 3:
                lam1 = abs(best[0])
                if abs(lam1 - 1.0) < 0.001:
                    print(f"    lambda_1={lam1:.8f} ~ 1 OK. "
                          f"Using {len(best)} converged eigenvalues.")
                    return [{'real': float(ev.real), 'imag': float(ev.imag),
                             'abs': float(abs(ev))} for ev in best[:3]]
                else:
                    print(f"    lambda_1={lam1:.8f} not ~ 1. "
                          f"Continuing attempts for better convergence.")
        except Exception as e:
            print(f"    WARNING: eigenvalue failed: {e}")
            break

    # return whatever we have, even if < 3
    if best:
        print(f"    Returning {len(best)} eigenvalue(s) after all attempts.")
        return [{'real': float(ev.real), 'imag': float(ev.imag),
                 'abs': float(abs(ev))} for ev in best[:3]]
    print("    No eigenvalues recovered.")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',           type=int,  required=True)
    parser.add_argument('--hours',           type=int,  required=True)
    parser.add_argument('--k_grid',          type=int,  required=True)
    parser.add_argument('--file_f',          type=str,  required=True)
    parser.add_argument('--file_cyc',        type=str,  required=True)
    parser.add_argument('--stage1_dir',      type=str,  required=True)
    parser.add_argument('--stage2_dir',      type=str,  required=True)
    parser.add_argument('--output_dir',      type=str,  required=True)
    parser.add_argument('--pairs',           nargs='+', required=True)
    parser.add_argument('--M_values',        nargs='+', type=int, default=[30])
    parser.add_argument('--r_grids',         type=int,  default=1000)
    parser.add_argument('--all_cycles_good', type=str,  default='false')
    parser.add_argument('--n_eigs',          type=int,  default=3,
                        help='Number of eigenvalues to compute (default 3)')
    args = parser.parse_args()
    tag = 'allcyclesaregood' if args.all_cycles_good.lower() == 'true' else ''

    os.makedirs(args.output_dir, exist_ok=True)

    data_list = np.load(args.file_f, allow_pickle=True).item()
    R = args.r_grids

    ##################### check here!!!!!!!!!!!!!!!!!
    data_list = {k: v[:720 * 6 * R] if isinstance(v, list) else v
                 for k, v in data_list.items()}
    print(f"  Points after slicing: {len(data_list['all_points'])}")

    cycles         = np.load(args.file_cyc, allow_pickle=True)
    movie_sequence = get_cycle_points(cycles, data_list)
    print(f"  Loaded {len(movie_sequence)} cycles.")

    periods_all = [len(movie_sequence['movie_points'].values[i])
                   for i in range(len(movie_sequence))]

    print("Computing U_k …")
    points        = np.vstack(movie_sequence['movie_points'].values).astype(float)
    U_k_all_dims  = compute_uk(points, args.k_grid)
    first_singleton_index = singleton(U_k_all_dims)
    print(f"  First singleton index: {first_singleton_index}")

    # ── option 2: compute eigenvalues once per k*, copy to all other pairs ────
    # P_k depends only on cycles + U_k, NOT on coord_idx/coord_idy.
    # So eigenvalues are identical for all self-pairs at the same k*.
    # Strategy:
    #   1. Collect all (M, k*) combinations needed across all pairs.
    #   2. For each unique (M, k*): compute once using the first pair, save.
    #   3. Copy the result JSON to all other pairs that share the same (M, k*).

    # Step 1: collect all (M, k*, cycle_ids) needed, grouped by (M, k*)
    # mk_to_pairs[(M, k*)] = list of (px, py) that need this combination
    mk_to_info  = {}   # (M, k*) -> {cycle_ids, periods}
    mk_to_pairs = {}   # (M, k*) -> [(px, py), ...]

    for pair_str in args.pairs:
        px, py = [int(x) for x in pair_str.split(',')]

        stage1_path = os.path.join(args.stage1_dir,
                                   f'bestcycles_idx{px}_idy{py}_r{args.r_grids}_{tag}.json')
        if not os.path.exists(stage1_path):
            continue

        for M in args.M_values:
            stage2_path = os.path.join(
                args.stage2_dir,
                f'plateaux_gr{args.group}_idx{px}_idy{py}_M{M}_r{args.r_grids}_{tag}.json'
            )
            if not os.path.exists(stage2_path):
                continue
            with open(stage2_path) as f:
                stage2 = json.load(f)
            K_intersection = stage2.get('K_intersection', [])
            cycle_ids      = stage2['cycle_ids']
            if not K_intersection:
                continue
            periods = [periods_all[cid] for cid in cycle_ids]
            for k in K_intersection:
                key = (M, k)
                if key not in mk_to_pairs:
                    mk_to_pairs[key] = []
                    mk_to_info[key]  = {'cycle_ids': cycle_ids, 'periods': periods}
                mk_to_pairs[key].append((px, py))

    print(f"\nUnique (M, k*) combinations to compute: {len(mk_to_info)}")

    # Step 2: for each unique (M, k*), compute once
    for (M, k_star_val), info in sorted(mk_to_info.items()):
        pairs_for_this_k = mk_to_pairs[(M, k_star_val)]
        cycle_ids        = info['cycle_ids']
        periods          = info['periods']

        # canonical filename: use first pair as representative
        px_ref, py_ref = pairs_for_this_k[0]
        ref_path = os.path.join(
            args.output_dir,
            f"eigenvalues_idx{px_ref}_idy{py_ref}_M{M}_k{k_star_val}"
            f"_gr{args.group}_{args.hours}h_r{args.r_grids}_{tag}.json"
        )

        # check if ANY pair already has this eigenvalue file computed
        computed_path = None
        for px, py in pairs_for_this_k:
            candidate = os.path.join(
                args.output_dir,
                f"eigenvalues_idx{px}_idy{py}_M{M}_k{k_star_val}"
                f"_gr{args.group}_{args.hours}h_r{args.r_grids}_{tag}.json"
            )
            if os.path.exists(candidate):
                computed_path = candidate
                break

        if computed_path is None:
            # compute P_k and eigenvalues once
            print(f"\n  Computing eigenvalues for M={M}, k*={k_star_val} "
                  f"(shared by {len(pairs_for_this_k)} pairs) ...")
            U_k = U_k_all_dims[k_star_val]
            print(f"  |U_k| = {len(U_k)}")
            try:
                P_k_csr, shape_P, N_gamma_map, E_gamma_map = build_Pk(
                    U_k, movie_sequence, cycle_ids, periods
                )
            except Exception as e:
                print(f"  ERROR building P_k: {e}. Skipping.")
                continue

            eigenvalues = compute_eigenvalues(P_k_csr, shape_P, n_eigs=args.n_eigs)
            ev_data = {
                'pair':        f"{px_ref},{py_ref}",
                'M':           M,
                'cycle_ids':   cycle_ids,
                'k_star':      k_star_val,
                'shape_P':     shape_P,
                'eigenvalues': eigenvalues,
            }
            with open(ref_path, 'w') as f:
                json.dump(ev_data, f, indent=2)
            print(f"  Eigenvalues computed → {ref_path}")
            computed_path = ref_path
        else:
            print(f"  M={M}, k*={k_star_val}: loading existing → {computed_path}")

        # Step 3: copy to all other pairs that need this (M, k*)
        with open(computed_path) as f:
            ev_data = json.load(f)

        for px, py in pairs_for_this_k:
            dest = os.path.join(
                args.output_dir,
                f"eigenvalues_idx{px}_idy{py}_M{M}_k{k_star_val}"
                f"_gr{args.group}_{args.hours}h_r{args.r_grids}_{tag}.json"
            )
            if os.path.exists(dest):
                continue
            # write with updated pair field
            ev_copy = dict(ev_data)
            ev_copy['pair'] = f"{px},{py}"
            with open(dest, 'w') as f:
                json.dump(ev_copy, f, indent=2)
        print(f"  Copied to {len(pairs_for_this_k)} pair files.")


if __name__ == '__main__':
    main()
