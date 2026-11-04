"""Table 1 (sampling frames), Table 2 (model serving windows), and the
Section 3.5 dependence numbers.

    output/table1_samples.csv
    output/table2_serving_windows.csv
    output/sec35_dependence.csv
"""
import pandas as pd

from common import (CORE6, CRIT, EVENT_FRAME, TREND_FRAME, W_BASE, W_EVENT,
                    W_POST, col, event_frame, icc, trend_frame, win, write)


def coverage_gaps(df):
    """Calendar days inside the frame's span carrying no observations."""
    days = pd.date_range(df.date.min(), df.date.max(), freq="D")
    missing = sorted(set(days) - set(df.date.unique()))
    runs, start, prev = [], None, None
    for d in missing:
        if start is None:
            start = prev = d
        elif (d - prev).days == 1:
            prev = d
        else:
            runs.append((start, prev))
            start = prev = d
    if start is not None:
        runs.append((start, prev))
    return len(days), len(missing), runs


def main():
    E, T = event_frame(), trend_frame()

    # ------------------------------------------------------------- Table 1
    rows = []
    for name, df in (("Trend sample", T), ("Event sample", E)):
        total, nmiss, runs = coverage_gaps(df)
        biggest = sorted(runs, key=lambda r: -((r[1] - r[0]).days + 1))[:3]
        rows.append({
            "sample": name,
            "turns": len(df),
            "clusters": df.cluster_id.nunique(),
            "conversations": df.conv_id.nunique(),
            "senders": df.sender_id.nunique(),
            "first_date": df.date.min().date(),
            "last_date": df.date.max().date(),
            "days_spanned": total,
            "empty_days": nmiss,
            "largest_gaps": "; ".join(
                f"{a.date()}-{b.date()} ({(b - a).days + 1}d)" for a, b in biggest),
        })
    write(pd.DataFrame(rows), "table1_samples.csv")

    # ------------------------------------------------------------- Table 2
    # Serving windows come from the trend sample, which spans the whole corpus.
    g = T.groupby("model").agg(n=("turn_id", "size"),
                               first_observed=("date", "min"),
                               last_observed=("date", "max"))
    g = g.sort_values("first_observed").reset_index()
    g["first_observed"] = g.first_observed.dt.date
    g["last_observed"] = g.last_observed.dt.date
    write(g, "table2_serving_windows.csv")

    # ------------------------------------------- Section 3.5 dependence
    # Window counts, cluster structure and per-criterion ICC. The paper reports
    # ICC as a RANGE across criteria, not a single value: several criteria sit
    # at a boundary in the event window and a single number is not reproducible.
    rows = []
    for label, w in (("baseline Apr15-23", W_BASE),
                     ("event Apr25-27", W_EVENT),
                     ("post May1-14", W_POST)):
        d = win(E, w)
        sizes = d.groupby("cluster_id").size()
        iccs = [icc(d, col(c)) for c in CORE6 + ["Holistic (1-5)"]]
        rows.append({
            "window": label,
            "turns": len(d),
            "clusters": d.cluster_id.nunique(),
            "conversations": d.conv_id.nunique(),
            "senders": d.sender_id.nunique(),
            "turns_per_cluster": round(sizes.mean(), 2),
            "largest_cluster": int(sizes.max()),
            "icc_min": round(min(iccs), 2),
            "icc_max": round(max(iccs), 2),
            "templated_share_pct": round(100 * d.templated.mean(), 1),
        })
    write(pd.DataFrame(rows), "sec35_dependence.csv")

    # The conversation-hash collision that forces the cluster unit.
    #
    # Scope matters here and the draft's §3.5 sentence is ambiguous about it.
    # Across the whole event sample the largest hash holds 583 turns from 583
    # senders; the 527 the draft quotes is the largest hash inside the
    # post-rollback window alone. Both rows are emitted so the claim can be
    # stated at whichever scope is intended.
    per_hash_senders = E.groupby("conv_id").sender_id.nunique()
    rows = []
    for label, d in ([("event sample", E)]
                     + [(l, win(E, w)) for l, w in (("baseline Apr15-23", W_BASE),
                                                    ("event Apr25-27", W_EVENT),
                                                    ("post May1-14", W_POST))]):
        counts = d.conv_id.value_counts()
        biggest = counts.idxmax()
        senders = d.groupby("conv_id").sender_id.nunique()
        rows.append({
            "scope": label,
            "hashes_with_5plus_senders": int((senders >= 5).sum()),
            "largest_hash_turns": int(counts.max()),
            "largest_hash_senders": int(senders.loc[biggest]),
        })
    write(pd.DataFrame(rows), "sec35_hash_collisions.csv")


if __name__ == "__main__":
    main()
