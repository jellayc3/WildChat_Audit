"""Table 6 (event window, all traffic) and Table 7 (the same three windows with
templated conversations removed).

One estimator, two traffic arms: OLS on a window indicator with cluster-robust
standard errors on conversation x sender. Cohen's d comes from turn-level
dispersion and is invariant to the clustering choice; only the intervals move.

Benjamini-Hochberg is applied within two families of seven -- the seven event
contrasts and the seven post contrasts -- which is the 14 the methods section
declares. The two traffic arms are corrected separately, because a reader of
either table is looking at one set of decisions, not both.

    output/table6_event_contrasts.csv
    output/table7_contrasts_templated_removed.csv
    output/table6_bootstrap_permutation.csv
"""
import pandas as pd

from common import (BH_FAMILY_EVENT, CRIT, W_BASE, W_EVENT, W_POST, bh,
                    cluster_bootstrap, cluster_permutation, clustered_contrast,
                    col, event_frame, traffic, win, write)


def contrast_table(df, label):
    base, event, post = win(df, W_BASE), win(df, W_EVENT), win(df, W_POST)
    rows = []
    for name in CRIT:
        c = col(name)
        ev = clustered_contrast(base, event, c)
        po = clustered_contrast(base, post, c)
        rows.append({
            "criterion": name,
            "pre": round(base[c].mean(), 2),
            "event": round(event[c].mean(), 2),
            "post": round(post[c].mean(), 2),
            "event_d": round(ev["d"], 4), "event_lo": round(ev["lo"], 4),
            "event_hi": round(ev["hi"], 4), "event_p": ev["p"],
            "event_deff": round(ev["deff"], 2),
            "post_d": round(po["d"], 4), "post_lo": round(po["lo"], 4),
            "post_hi": round(po["hi"], 4), "post_p": po["p"],
            "post_deff": round(po["deff"], 2),
        })
    T = pd.DataFrame(rows)

    # Two families of seven. Assert the size rather than trusting it.
    assert len(T) * 2 == BH_FAMILY_EVENT, f"BH family is {len(T)*2}, expected {BH_FAMILY_EVENT}"
    T["event_p_bh"] = bh(T.event_p)
    T["post_p_bh"] = bh(T.post_p)

    T.attrs["n"] = (len(base), len(event), len(post))
    print(f"  {label}: n = {len(base)} / {len(event)} / {len(post)} turns")
    return T


def main():
    E = event_frame()

    t6 = contrast_table(traffic(E, "pooled"), "all traffic")
    write(t6, "table6_event_contrasts.csv")

    t7 = contrast_table(traffic(E, "templated_removed"), "templated removed")
    write(t7, "table7_contrasts_templated_removed.csv")

    # Window counts as their own artifact: the tables print them as a footer
    # row and the verifier checks them separately.
    counts = []
    for arm in ("pooled", "templated_removed"):
        d = traffic(E, arm)
        counts.append({"traffic": arm,
                       "pre": len(win(d, W_BASE)),
                       "event": len(win(d, W_EVENT)),
                       "post": len(win(d, W_POST))})
    write(pd.DataFrame(counts), "table67_window_counts.csv")

    # The other two legs of the three-estimator argument, on all traffic.
    # Reported so the paper can state that the three agree rather than
    # asserting it.
    base, event = win(E, W_BASE), win(E, W_EVENT)
    rows = []
    for name in CRIT:
        c = col(name)
        lo, hi = cluster_bootstrap(base, event, c)
        rows.append({"criterion": name,
                     "bootstrap_lo": round(lo, 3), "bootstrap_hi": round(hi, 3),
                     "permutation_p": cluster_permutation(base, event, c)})
    B = pd.DataFrame(rows)
    B["permutation_p_bh"] = bh(B.permutation_p)
    write(B, "table6_bootstrap_permutation.csv")


if __name__ == "__main__":
    main()
