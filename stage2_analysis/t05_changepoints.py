"""Table 5 (consensus changepoint weeks, all traffic) and Table 8 (the same
detector with templated conversations removed).

One changepoint analysis, declared once in config: W-MON bins labelled by the
date they end, every non-empty week, PELT with an l2 cost and a BIC-style
penalty log(n)*var(diff)/2. The bin labelled 2025-04-28 covers April 22-27.

    output/table5_changepoints_all.csv
    output/table8_changepoints_templated_removed.csv
    output/sec412_weekly_series.csv
"""
import pandas as pd

from common import (CONSENSUS_K, CORE6, bootstrap_support, changepoints,
                    consensus_weeks, event_frame, traffic, write)


def series(df, label):
    M, cps = changepoints(df)
    cons = consensus_weeks(cps)
    support = bootstrap_support(df, cps)

    rows = []
    for week, k in cons.items():
        flagged = [c for c in CORE6 if week in cps.get(c, [])]
        # A W-MON bin labelled D contains timestamps in (D-7, D]. The observed
        # range can be shorter where the corpus has no data: the bin labelled
        # April 28 spans April 22-28 but is observed only through April 27,
        # because WildChat carries nothing for April 28-30.
        lo = week - pd.Timedelta(days=6)
        obs = df[(df.date >= lo) & (df.date <= week)].date
        rows.append({
            "week_end": week.date(),
            "bin_covers": f"{lo.date()} to {week.date()}",
            "observed": (f"{obs.min().date()} to {obs.max().date()}"
                         if len(obs) else "no data"),
            "n_criteria": k,
            "criteria": ", ".join(flagged),
            "bootstrap_support_pct": ", ".join(
                f"{support[c][week]:.0f}%" for c in flagged),
        })
    T = pd.DataFrame(rows)
    print(f"  {label}: {len(M)} weekly bins, {len(cons)} consensus weeks "
          f"(>= {CONSENSUS_K} of {len(CORE6)})")
    return T, M


def main():
    E = event_frame()

    t5, M_all = series(traffic(E, "pooled"), "all traffic")
    write(t5, "table5_changepoints_all.csv")

    t8, M_ord = series(traffic(E, "templated_removed"), "templated removed")
    write(t8, "table8_changepoints_templated_removed.csv")

    # The weekly series both tables sit on, plus the templated share, so the
    # figures and the Supplement A week-context notes read from one file.
    W = M_all.add_suffix("_all").join(M_ord.add_suffix("_ordinary"), how="outer")
    share = (E.set_index("ts").templated.resample("W-MON").mean() * 100).rename("templated_share_pct")
    n = E.set_index("ts").turn_id.resample("W-MON").count().rename("turns")
    W = W.join(share).join(n)
    W = W[W.turns > 0].round(4)
    W.index.name = "week_end"
    write(W.reset_index(), "sec412_weekly_series.csv")

    shared = set(t5.week_end) & set(t8.week_end)
    print(f"  weeks in both series: {sorted(shared)}")


if __name__ == "__main__":
    main()
