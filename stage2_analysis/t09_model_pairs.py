"""Table 9: how consistently model identity separates concurrently served pairs,
before and after correction, on all traffic and with templated conversations
removed. Plus the per-pair detail and the Section 4.2.1 focus pair.

Pairs qualify on the trend sample, which spans the whole corpus and therefore
carries the concurrency. A pair qualifies when the two serving windows overlap
by at least PAIR_MIN_OVERLAP_DAYS and each model contributes at least
PAIR_MIN_TURNS inside the overlap. The o1 models are excluded as a different
product class.

Benjamini-Hochberg is applied across the whole 7 x 8 = 56 family.

    output/table9_model_pairs.csv
    output/table9_pairs_detail.csv
    output/sec421_focus_pair.csv
"""
import itertools

import pandas as pd

from common import (BH_FAMILY_PAIRS, CRIT, PAIR_EXCLUDE, PAIR_MIN_OVERLAP_DAYS,
                    PAIR_MIN_TURNS, bh, cluster_permutation, clustered_contrast,
                    col, traffic, trend_frame, write)


def serving_windows(T):
    # Overlap is measured on TIMESTAMPS, not on dates normalised to midnight.
    # The distinction matters: normalising extends each window to the whole of
    # its final calendar day and pulls in turns served after the other model had
    # already stopped, which shifts every effect size in the last day of an
    # overlap.
    g = T.groupby("model").agg(first=("ts", "min"), last=("ts", "max"),
                               n=("turn_id", "size"))
    return g[~g.index.isin(PAIR_EXCLUDE)]


def qualifying_pairs(T, min_overlap=PAIR_MIN_OVERLAP_DAYS, min_turns=PAIR_MIN_TURNS,
                     windows_from=None):
    """Every model pair whose serving windows overlap enough, with the overlap
    window itself so the contrast is run only inside it.

    `windows_from` is the frame the serving windows are measured on, which is
    always the full sample even when the contrast runs on filtered traffic: when
    a model was being served is a property of the deployment, not of which
    turns a traffic treatment happens to keep. Recomputing the windows on the
    filtered frame would let a traffic treatment silently move the comparison
    period as well as its contents.
    """
    g = serving_windows(T if windows_from is None else windows_from)
    out = []
    for a, b in itertools.combinations(g.index, 2):
        lo = max(g.loc[a, "first"], g.loc[b, "first"])
        hi = min(g.loc[a, "last"], g.loc[b, "last"])
        if (hi - lo).days < min_overlap:
            continue
        seg = T[(T.ts >= lo) & (T.ts <= hi)]
        na = (seg.model == a).sum()
        nb = (seg.model == b).sum()
        if na < min_turns or nb < min_turns:
            continue
        out.append((a, b, lo, hi, int(na), int(nb)))
    return out


def pair_contrasts(T, arm, min_overlap=PAIR_MIN_OVERLAP_DAYS,
                   min_turns=PAIR_MIN_TURNS, unit=None, permute=False):
    """Every criterion on every qualifying pair, BH-corrected across the family."""
    d = traffic(T, arm)
    pairs = qualifying_pairs(d, min_overlap, min_turns, windows_from=T)
    rows = []
    for a, b, lo, hi, na, nb in pairs:
        seg = d[(d.ts >= lo) & (d.ts <= hi)]
        A, B = seg[seg.model == a], seg[seg.model == b]
        for name in CRIT:
            c = col(name)
            kw = {"unit": unit} if unit else {}
            r = clustered_contrast(B, A, c, **kw)
            # n reported as the non-null count for THIS criterion, which is
            # what the contrast actually ran on.
            row = {"pair": f"{a} vs {b}", "model_a": a, "model_b": b,
                   "overlap_start": lo.date(), "overlap_end": hi.date(),
                   "n_a": int(A[c].notna().sum()), "n_b": int(B[c].notna().sum()),
                   "criterion": name,
                   "d": r["d"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]}
            if permute:
                row["perm_p"] = cluster_permutation(B, A, c)
            rows.append(row)
    R = pd.DataFrame(rows)
    if R.empty:
        return R, pairs
    R["p_bh"] = bh(R.p)
    if permute:
        R["perm_p_bh"] = bh(R.perm_p)
    return R, pairs


def main():
    T = trend_frame()

    detail = {}
    for arm in ("pooled", "templated_removed"):
        R, pairs = pair_contrasts(T, arm, permute=(arm == "pooled"))
        detail[arm] = R
        print(f"  {arm}: {len(pairs)} qualifying pairs, {len(R)} tests")
        if arm == "pooled":
            assert len(R) == BH_FAMILY_PAIRS, \
                f"BH family is {len(R)}, expected {BH_FAMILY_PAIRS}"

    # ------------------------------------------------------------- Table 9
    rows = []
    for name in CRIT:
        r = {"criterion": name}
        for arm, tag in (("pooled", "all_traffic"),
                         ("templated_removed", "templated_removed")):
            d = detail[arm]
            d = d[d.criterion == name]
            r[f"{tag}_uncorrected"] = int((d.p < 0.05).sum())
            r[f"{tag}_bh"] = int((d.p_bh < 0.05).sum())
            r[f"{tag}_n_pairs"] = len(d)
        rows.append(r)
    t9 = pd.DataFrame(rows)
    write(t9, "table9_model_pairs.csv")
    print(t9.to_string(index=False))

    # ------------------------------------------------------- per-pair detail
    both = pd.concat([detail["pooled"].assign(traffic="all"),
                      detail["templated_removed"].assign(traffic="templated_removed")],
                     ignore_index=True)
    write(both.round(4), "table9_pairs_detail.csv")

    # Per-pair separation counts, the supplement's S5 table.
    # n_a / n_b vary by criterion (non-null counts), so aggregate those with
    # min; grouping on them would split each pair into one row per criterion.
    counts = (detail["pooled"].assign(sig=lambda d: d.p_bh < 0.05,
                                      sig_perm=lambda d: d.perm_p_bh < 0.05)
              .groupby("pair", sort=False)
              .agg(overlap_start=("overlap_start", "first"),
                   overlap_end=("overlap_end", "first"),
                   n_a=("n_a", "min"), n_b=("n_b", "min"),
                   clustered=("sig", "sum"), permutation=("sig_perm", "sum"),
                   max_abs_d=("d", lambda s: s.abs().max()))
              .reset_index().sort_values("max_abs_d", ascending=False))
    write(counts.round(3), "s5_pairs_per_pair.csv")

    # ------------------------------------------------------ 4.2.1 focus pair
    focus = detail["pooled"][
        detail["pooled"].pair.str.contains("gpt-4o-2024-11-20")].copy()
    if not focus.empty:
        ordinary = detail["templated_removed"]
        ordinary = ordinary[ordinary.pair.isin(focus.pair.unique())][
            ["pair", "criterion", "d", "p_bh"]].rename(
            columns={"d": "d_templated_removed", "p_bh": "p_bh_templated_removed"})
        focus = focus.merge(ordinary, on=["pair", "criterion"], how="left")
    write(focus.round(4), "sec421_focus_pair.csv")
    if not focus.empty:
        print("\n  4.2.1 focus pair:")
        print(focus[["pair", "criterion", "d", "p_bh", "perm_p_bh",
                     "d_templated_removed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
