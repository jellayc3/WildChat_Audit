"""Section 4.2.3 and 4.3: what the criteria correlate with.

Three separate things, kept in separate files because they run on different
frames and the paper quotes them in different sections:

  sec43_length_correlations   Pearson r against log reply length, EVENT sample.
                              This is the set the paper prints in 4.3 and in
                              Table 10's last row.
  sec423_weekly_templated     Spearman rho between each criterion's weekly mean
                              and the weekly templated share, EVENT sample,
                              across the 37 weekly bins.
  sec423_criterion_matrix     Inter-criterion Spearman correlations in both
                              samples, which is Figure 2.

    output/sec43_length_correlations.csv
    output/sec423_weekly_templated.csv
    output/sec423_criterion_correlations_{event,trend}.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import CRIT, col, event_frame, trend_frame, write


def main():
    E, T = event_frame(), trend_frame()

    # ------------------------------------------------- length correlations
    rows = []
    for name in CRIT:
        c = col(name)
        d = E[[c, "uw", "aw"]].dropna()
        rows.append({
            "criterion": name,
            "r_log_assistant_words": stats.pearsonr(d[c], np.log1p(d.aw))[0],
            "r_log_user_words": stats.pearsonr(d[c], np.log1p(d.uw))[0],
            "variance_explained_assistant_pct":
                100 * stats.pearsonr(d[c], np.log1p(d.aw))[0] ** 2,
            "n": len(d),
        })
    L = pd.DataFrame(rows).round(4)
    write(L, "sec43_length_correlations.csv")
    print("  length correlations (event sample):")
    print(L[["criterion", "r_log_assistant_words", "r_log_user_words"]]
          .round(2).to_string(index=False))

    # --------------------------------------- weekly templated-share rho
    B = E.set_index("ts")
    weekly = pd.DataFrame({
        name: B[col(name)].resample("W-MON").mean() for name in CRIT})
    weekly["templated_share"] = B.templated.resample("W-MON").mean()
    weekly["turns"] = B.turn_id.resample("W-MON").count()
    weekly = weekly[weekly.turns > 0].dropna()

    rows = []
    for name in CRIT:
        rho, p = stats.spearmanr(weekly[name], weekly.templated_share)
        rows.append({"criterion": name, "spearman_rho": round(rho, 4),
                     "p": p, "n_weeks": len(weekly)})
    W = pd.DataFrame(rows)
    write(W, "sec423_weekly_templated.csv")
    print(f"\n  weekly rho against templated share ({len(weekly)} bins):")
    print(W.round(4).to_string(index=False))

    # ------------------------------------------ inter-criterion matrix
    for label, df in (("event", E), ("trend", T)):
        M = df[[col(n) for n in CRIT]].rename(
            columns={col(n): n for n in CRIT}).corr(method="spearman").round(3)
        M.index.name = "criterion"
        write(M.reset_index(), f"sec423_criterion_correlations_{label}.csv")


if __name__ == "__main__":
    main()
