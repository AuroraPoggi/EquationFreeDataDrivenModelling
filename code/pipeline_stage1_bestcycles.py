"""
Stage 1 (all-cycles-good variant introduced in paper in Section 2.9) 

The full cycle-selection procedure (d_gamma / g_gamma outlier removal), found that for the datasets used in this
paper every detected cycle with period > 1 already qualifies as "good".

Usage:
  python pipeline_stage1_bestcycles.py \\
      --file_cyc  24hmov_6hours_gr0_unique_cycles_basinmovies_period_k5_s6_r1000.npy \\
      --pairs_file pairs_list.txt \\
      --stage1_dir  pipeline/gr0_r1000/stage1_bestcycles \\
      --r_grids 1000
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file_cyc", required=True, help="Output of main_getcycles_countpointsinbasin.py.")
    parser.add_argument("--pairs_file", required=True, help="Text file, one 'px,py' projection pair per line.")
    parser.add_argument("--stage1_dir", required=True, help="Output directory for the per-pair JSON files.")
    parser.add_argument("--r_grids", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.stage1_dir, exist_ok=True)

    cycles = np.load(args.file_cyc, allow_pickle=True)
    all_cycle_ids = [i for i, c in enumerate(cycles) if len(c) > 1]
    print(f"Valid cycles: {len(all_cycle_ids)}")

    with open(args.pairs_file) as f:
        pairs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    for pair_str in pairs:
        if not pair_str or "," not in pair_str:
            continue
        px, py = [int(x.strip()) for x in pair_str.split(",")]
        result = {
            "projection_x": px,
            "projection_y": py,
            "good_cycles": all_cycle_ids,
            "best_box_cycle": all_cycle_ids[0] if all_cycle_ids else None,
            "g_gamma": {str(cid): 0.0 for cid in all_cycle_ids},
            "cycle_sets": [all_cycle_ids],
        }
        out_path = os.path.join(
            args.stage1_dir, f"bestcycles_idx{px}_idy{py}_r{args.r_grids}_allcyclesaregood.json"
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Written -> {out_path}")

    print("Stage 1 complete.")
