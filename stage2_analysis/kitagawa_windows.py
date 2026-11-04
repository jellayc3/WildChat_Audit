"""Kitagawa decomposition across the four April windows, by any factor.

Generalises kitagawa_templated.py from templated/ordinary to an arbitrary
grouping factor, and runs it on the event frame and both ordinary draws.

    total  = y2 - y1
    mix    = sum_g (s2_g - s1_g) * ybar_g      ybar_g = mean(y1_g, y2_g)
    within = sum_g sbar_g * (y2_g - y1_g)      sbar_g = (s1_g + s2_g) / 2

The algebra is unchanged. What is added is COMMON SUPPORT,

    support = sum_g min(s1_g, s2_g)

the overlap between the two windows' group distributions. A group present in
only one window contributes to `mix` by construction and can contribute nothing
to `within`, because the data hold no counterfactual for it. Across the April 15
serving change the two windows share almost no served model, so the model term
is arithmetic rather than evidence; the flag makes that explicit instead of
letting a large "mix" share be read as attribution.

    output/kitagawa_windows_summary.csv
    output/kitagawa_windows_detail.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (KIT_MIN_SUPPORT, KIT_WINDOWS, ORDINARY_FRAME,
                    col, event_frame, write)

COMPARISONS = [
    ("W1 Apr 6-14 (pre-switch)",   "W2 Apr 15-23 (post-switch)", "the model switch"),
    ("W2 Apr 15-23 (post-switch)", "W3 Apr 25-28 (event)",       "the disclosed event"),
    ("W3 Apr 25-28 (event)",       "W4 May 1-8 (post-event)",    "after the rollback"),
    ("W1 Apr 6-14 (pre-switch)",   "W4 May 1-8 (post-event)",    "end to end"),
]
FACTORS = [("model", "served model"), ("traffic", "templated vs ordinary"),
           ("lang5", "language"), ("len3", "reply-length tercile"),
           ("cell", "traffic x language x length")]


def kitagawa(pre, post, column, gcol):
    n1, n2 = len(pre), len(post)
    groups = sorted(set(pre[gcol].dropna()) | set(post[gcol].dropna()), key=str)
    mix = within = support = 0.0
    rows = []
    for g in groups:
        p1 = pre.loc[pre[gcol] == g, column].dropna()
        p2 = post.loc[post[gcol] == g, column].dropna()
        s1 = len(p1) / n1 if n1 else 0.0
        s2 = len(p2) / n2 if n2 else 0.0
        y1 = p1.mean() if len(p1) else np.nan
        y2 = p2.mean() if len(p2) else np.nan
        if np.isnan(y1) and np.isnan(y2):
            continue
        ybar = float(np.nanmean([y1, y2]))
        dy = 0.0 if (np.isnan(y1) or np.isnan(y2)) else float(y2 - y1)
        sbar = (s1 + s2) / 2
        m, w = (s2 - s1) * ybar, sbar * dy
        mix += m
        within += w
        if len(p1) and len(p2):
            support += min(s1, s2)
        rows.append(dict(group=str(g), s1=s1, s2=s2, y1=y1, y2=y2,
                         mix=m, within=w, contrib=m + w,
                         in_both=bool(len(p1) and len(p2))))
    total = post[column].dropna().mean() - pre[column].dropna().mean()
    return dict(total=total, mix=mix, within=within,
                resid=total - mix - within, support=support), rows


def add_factors(df):
    """Language collapsed to the five largest over the decomposition span, and
    reply length in terciles cut ONCE on that span so bins do not move between
    windows."""
    lo = KIT_WINDOWS["W1 Apr 6-14 (pre-switch)"][0]
    hi = KIT_WINDOWS["W4 May 1-8 (post-event)"][1]
    span = df[(df.date >= lo) & (df.date < hi)]
    top = list(span.language.value_counts().head(5).index)
    df = df.copy()
    df["lang5"] = np.where(df.language.isin(top), df.language, "Other")
    q = span.aw.quantile([1 / 3, 2 / 3]).values
    df["len3"] = np.where(df.aw <= q[0], "short",
                 np.where(df.aw <= q[1], "medium", "long"))
    df["traffic"] = np.where(df.templated, "templated", "ordinary")
    df["cell"] = df.traffic + "|" + df.lang5 + "|" + df.len3
    return df


def load_ordinary(draw):
    p = ORDINARY_FRAME[draw]
    if not os.path.exists(p):
        p = p.replace(".parquet", ".csv.gz")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p) if p.endswith(".csv.gz") else pd.read_parquet(p)
    d["date"] = pd.to_datetime(d.date)
    return d


def main():
    datasets = [("Event sample, all traffic (12B)", event_frame(),
                 col("Validation"))]
    for draw in ("A", "B"):
        d = load_ordinary(draw)
        if d is None:
            print(f"  ! ordinary draw {draw} frame missing; "
                  f"run stage1_frames/build_ordinary.py")
        else:
            datasets.append((f"Ordinary Draw {draw} (27B)", d, "validation"))

    summary, detail = [], []
    for name, df, column in datasets:
        df = add_factors(df)
        for w1, w2, story in COMPARISONS:
            a, b = KIT_WINDOWS[w1], KIT_WINDOWS[w2]
            pre = df[(df.date >= a[0]) & (df.date < a[1])]
            post = df[(df.date >= b[0]) & (df.date < b[1])]
            if pre.empty or post.empty:
                continue
            for gcol, fname in FACTORS:
                if gcol not in df or df[gcol].nunique() < 2:
                    continue
                r, rows = kitagawa(pre, post, column, gcol)
                ident = r["support"] >= KIT_MIN_SUPPORT
                summary.append(dict(
                    dataset=name, comparison=f"{w1} -> {w2}", story=story,
                    factor=fname, n_pre=len(pre), n_post=len(post),
                    total=r["total"], mix=r["mix"], within=r["within"],
                    resid=r["resid"],
                    mix_share=r["mix"] / r["total"] if abs(r["total"]) > 1e-9 else np.nan,
                    common_support=r["support"], identified=ident))
                for rr in rows:
                    detail.append(dict(dataset=name, comparison=f"{w1} -> {w2}",
                                       factor=fname, **rr))

    S = pd.DataFrame(summary)
    write(S, "kitagawa_windows_summary.csv")
    write(pd.DataFrame(detail), "kitagawa_windows_detail.csv")

    print()
    for name in S.dataset.unique():
        print(name)
        for cmp_ in S[S.dataset == name].comparison.unique():
            g = S[(S.dataset == name) & (S.comparison == cmp_)]
            print(f"  {cmp_}   total {g.total.iloc[0]:+.3f}")
            for _, r in g.iterrows():
                flag = "" if r.identified else "   NOT IDENTIFIED"
                print(f"    {r.factor:28s} mix {r.mix:+7.3f}  "
                      f"within {r.within:+7.3f}  support "
                      f"{r.common_support * 100:5.1f}%{flag}")
        print()


if __name__ == "__main__":
    main()
