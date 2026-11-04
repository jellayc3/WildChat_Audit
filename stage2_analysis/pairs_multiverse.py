"""The 54-specification concurrent-model-pair grid.

    3 overlap thresholds x 3 volume thresholds x 3 clustering units
      x 2 traffic treatments = 54

Separate from the event grid; the two counts never combine. Reports, per
criterion, how many of the qualifying pairs separate under each specification.

    output/pairs_multiverse.csv
    output/pairs_multiverse_detail.csv
"""
import itertools
import warnings

import pandas as pd

from common import (CRIT, GRID_PAIR_OVERLAP, GRID_PAIR_UNITS, GRID_PAIR_VOLUME,
                    GRID_TRAFFIC, N_SPECS_PAIRS, bh, clustered_contrast, col,
                    traffic, trend_frame, write)
from t09_model_pairs import qualifying_pairs

warnings.filterwarnings("ignore")

def main():
    T = trend_frame()
    rows = []
    spec = 0

    for overlap, volume, unit, arm in itertools.product(
            GRID_PAIR_OVERLAP, GRID_PAIR_VOLUME, GRID_PAIR_UNITS, GRID_TRAFFIC):
        spec += 1
        d = traffic(T, arm)
        pairs = qualifying_pairs(d, overlap, volume, windows_from=T)
        fits = []
        for a, b, lo, hi, _, _ in pairs:
            seg = d[(d.ts >= lo) & (d.ts <= hi)]
            A, B = seg[seg.model == a], seg[seg.model == b]
            for name in CRIT:
                r = clustered_contrast(B, A, col(name), unit=unit)
                fits.append({"criterion": name, "pair": f"{a} vs {b}", "p": r["p"],
                             "d": r["d"]})
        if not fits:
            continue
        F = pd.DataFrame(fits)
        F["p_bh"] = bh(F.p)
        for name in CRIT:
            sub = F[F.criterion == name]
            rows.append({
                "spec": spec, "overlap_days": overlap, "min_turns": volume,
                "unit": unit, "traffic": arm, "criterion": name,
                "n_pairs": len(sub),
                "pairs_separating": int((sub.p_bh < 0.05).sum()),
            })
        if spec % 9 == 0:
            print(f"  {spec} / {N_SPECS_PAIRS} specifications")

    M = pd.DataFrame(rows)
    n_spec = M.spec.nunique()
    print(f"  {n_spec} specifications")
    assert n_spec == N_SPECS_PAIRS, \
        f"pair grid is {n_spec} specifications, config says {N_SPECS_PAIRS}"
    write(M, "pairs_multiverse_detail.csv")

    S = (M.groupby("criterion")
         .agg(median_pairs=("pairs_separating", "median"),
              min_pairs=("pairs_separating", "min"),
              max_pairs=("pairs_separating", "max"),
              mean_pairs=("pairs_separating", "mean"))
         .reindex(list(CRIT)).reset_index().round(2))
    write(S, "pairs_multiverse.csv")
    print(S.to_string(index=False))

    # The paper's claim that removing templated turns raises the average
    # number of separating pairs.
    by_arm = M.groupby("traffic").pairs_separating.mean().round(2)
    write(by_arm.rename("mean_pairs_separating").reset_index(),
          "pairs_multiverse_by_traffic.csv")
    print("\n  mean separating pairs by traffic arm:")
    print(by_arm.to_string())

    every = M.groupby("spec").pairs_separating.min().gt(0).all()
    print(f"  every criterion separates >=1 pair in every specification: {bool(every)}")


if __name__ == "__main__":
    main()
