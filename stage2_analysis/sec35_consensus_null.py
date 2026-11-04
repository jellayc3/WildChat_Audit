"""The consensus-rule null in Section 3.5, and the p-value for the observed
number of consensus weeks.

The rule treats a week as a consensus changepoint when at least two of the six
core criteria flag it. The criteria are not independent -- validation and
positive affirmation correlate at rho = 0.84 -- so the reference distribution is
built by permuting week ORDER once and applying that same permutation to all six
columns, keeping each week's six values together. Permuting the criteria
separately would assume the correlation away and understate the false-positive
rate.

Slow: 2,000 draws x 6 PELT fits.

    output/sec35_consensus_null.csv
"""
import numpy as np
import pandas as pd

from common import (CONSENSUS_K, CORE6, NULL_DRAWS, changepoints,
                    consensus_null, consensus_weeks, event_frame, weekly_matrix,
                    write)


def main():
    E = event_frame()
    M = weekly_matrix(E)
    _, cps = changepoints(E)
    observed = consensus_weeks(cps)

    print(f"  {len(M)} weekly bins, {NULL_DRAWS} correlation-preserving permutations")
    rates, ncons = consensus_null(M)

    n_obs = len(observed)
    # P(at least as many consensus weeks as observed) under the null.
    p_series = float((ncons >= n_obs).mean())

    rows = []
    for k in (2, 3, 4):
        hit = sum(1 for v in observed.values() if v >= k)
        rows.append({
            "threshold": f">= {k} of 6 criteria",
            "null_rate_per_week_pct": round(rates[k], 2),
            "expected_weeks_per_series": round(rates[k] / 100 * len(M), 2),
            "observed_weeks": hit,
        })
    T = pd.DataFrame(rows)
    T.attrs["p_series"] = p_series
    write(T, "sec35_consensus_null.csv")

    summary = pd.DataFrame([{
        "weekly_bins": len(M),
        "consensus_k": CONSENSUS_K,
        "observed_consensus_weeks": n_obs,
        "expected_consensus_weeks": round(ncons.mean(), 2),
        "p_observed_or_more": round(p_series, 4),
        "draws": NULL_DRAWS,
    }])
    write(summary, "sec35_consensus_null_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
