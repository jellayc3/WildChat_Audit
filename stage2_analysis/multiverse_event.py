"""The 144-specification event grid (Simonsohn et al. 2020; Steegen et al. 2016).

Full factorial, no cell dropped:

    4 baseline windows x 2 event windows x 2 traffic treatments
      x 3 units of analysis x 3 length adjustments = 144

The three units are the turn, the CLUSTER mean and the first turn per cluster,
where the cluster is conversation x sender -- the same unit the standard errors
are clustered on, not the conversation.

The length covariate is log1p(words), taken at TURN level and, under the
cluster-mean unit, aggregated as the within-cluster mean of those logs.
Averaging raw word counts and logging afterwards would fit log(mean words), a
different covariate from the one the other two units use.

Only the two traffic definitions the paper reports are arms here. The retired
rules are not in the grid: dropping an arm because it gives a different answer
would defeat the purpose, but these were never reported as treatments at all.

    output/multiverse_event.csv
    output/multiverse_by_criterion.csv
    output/multiverse_by_traffic.csv
    output/multiverse_headline.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (CLUSTER_UNIT, UNIT_ID, CRIT, GRID_BASELINES, GRID_EVENTS,
                    GRID_MIN_N, N_SPECS, bh, cohens_d, col, event_frame, win,
                    write)

warnings.filterwarnings("ignore")

UNITS = ["turn", "cluster mean", "first turn per cluster"]
ADJUSTMENTS = {"none": [], "log user len": ["lu"], "log user+asst len": ["lu", "la"]}
TRAFFIC = {
    "none (pooled)":        lambda d: d,
    "excl templated (H&T)": lambda d: d[~d.templated],
}


def main():
    E = event_frame()
    rows = []

    for (bn, bw), (en, ew), (fn, ff), unit, (an, cov) in itertools.product(
            GRID_BASELINES.items(), GRID_EVENTS.items(), TRAFFIC.items(),
            UNITS, ADJUSTMENTS.items()):
        b, e = ff(win(E, bw)), ff(win(E, ew))
        if unit == "first turn per cluster":
            b = b.sort_values("assistant_turn_index").groupby(UNIT_ID).head(1)
            e = e.sort_values("assistant_turn_index").groupby(UNIT_ID).head(1)

        for name in CRIT:
            c = col(name)
            bb = b[[c, UNIT_ID, CLUSTER_UNIT, "uw", "aw"]].dropna().copy()
            ee = e[[c, UNIT_ID, CLUSTER_UNIT, "uw", "aw"]].dropna().copy()
            for f in (bb, ee):
                f["lu"] = np.log1p(f.uw)
                f["la"] = np.log1p(f.aw)
            if unit == "cluster mean":
                bb = bb.groupby([UNIT_ID, CLUSTER_UNIT], as_index=False)[[c, "lu", "la"]].mean()
                ee = ee.groupby([UNIT_ID, CLUSTER_UNIT], as_index=False)[[c, "lu", "la"]].mean()
            if len(bb) < GRID_MIN_N or len(ee) < GRID_MIN_N:
                continue

            d, sp = cohens_d(ee[c], bb[c])
            if not np.isfinite(d):
                continue
            pool = pd.concat([bb.assign(_e=0.0), ee.assign(_e=1.0)], ignore_index=True)
            X = pd.DataFrame({"_e": pool._e.values})
            for k in cov:
                X[k] = pool[k].values
            m = sm.OLS(pool[c].astype(float).values, sm.add_constant(X)).fit(
                cov_type="cluster", cov_kwds={"groups": pool[CLUSTER_UNIT].values})
            rows.append(dict(criterion=name, baseline=bn, event=en, traffic=fn,
                             unit=unit, adjust=an, n_base=len(bb), n_event=len(ee),
                             d=m.params["_e"] / sp, p=m.pvalues["_e"]))

    M = pd.DataFrame(rows)
    M["spec"] = M.baseline + "|" + M.event + "|" + M.traffic + "|" + M.unit + "|" + M.adjust
    M["p_bh"] = M.groupby("spec").p.transform(lambda s: bh(s.values))
    n_spec = M.spec.nunique()
    print(f"  {len(M)} fits across {n_spec} specifications")
    assert n_spec == N_SPECS, f"grid is {n_spec} specifications, config says {N_SPECS}"
    write(M.round(6), "multiverse_event.csv")

    flip = lambda s: 100 * min((s > 0).mean(), (s < 0).mean())

    by_traffic = (M.groupby("traffic")
                  .agg(pct_significant=("p_bh", lambda s: 100 * (s < 0.05).mean()),
                       mean_abs_d=("d", lambda s: s.abs().mean()), n=("p", "size"))
                  .reindex(list(TRAFFIC)))
    write(by_traffic.round(2).reset_index(), "multiverse_by_traffic.csv")
    print("\n  BH-significant by traffic definition:")
    print(by_traffic.round(2).to_string())

    by_crit = (M.groupby("criterion")
               .agg(median_d=("d", "median"), d_min=("d", "min"), d_max=("d", "max"),
                    pct_significant=("p_bh", lambda s: 100 * (s < 0.05).mean()),
                    pct_sign_flip=("d", flip))
               .reindex(list(CRIT)))
    write(by_crit.round(2).reset_index(), "multiverse_by_criterion.csv")
    print("\n  stability by criterion:")
    print(by_crit.round(2).to_string())

    head = pd.DataFrame([{
        "n_specifications": n_spec,
        "pct_significant_pooled": round(by_traffic.loc["none (pooled)", "pct_significant"], 1),
        "pct_significant_templated_removed": round(
            by_traffic.loc["excl templated (H&T)", "pct_significant"], 1),
        "holistic_pct_sign_flip": round(by_crit.loc["Holistic (1-5)", "pct_sign_flip"], 1),
        "disagree_pct_sign_flip": round(by_crit.loc["Disagree", "pct_sign_flip"], 1),
    }])
    write(head, "multiverse_headline.csv")
    print("\n" + head.T.to_string(header=False))


if __name__ == "__main__":
    main()
