"""Kitagawa (mix vs within) decomposition by templated / ordinary traffic.

Same algebra as the Nov 18 / Dec 2 model-mix split in the WildChat audit paper,
but groups are templated vs not-templated instead of model string:

    total = post_mean - pre_mean = mix + within
    mix    = sum_g (s2_g - s1_g) * ybar_g     # traffic-mix term
    within = sum_g sbar_g * (y2_g - y1_g)     # within-group behavior term

Because the two groups can move in opposite directions, the headline
"bot-traffic share of the jump" is best read as the templated group's
within-term share of the pooled total (sbar_T * dy_T / total), not the
aggregate mix term alone. When pre-window bots score low and event-window
bots score high, ybar_T ≈ ybar_O and the mix term collapses even though
the jump is almost entirely bot-driven.

Comparisons
-----------
1. Apr 15-23 vs Apr 25-27   (Table 6 event contrast)
2. Apr 8-14  vs Apr 15-23   (model-transition contrast)
3. Each Table 5 consensus week, +/- 14 days around the week-end label

    output/kitagawa_templated_windows.csv
    output/kitagawa_templated_changepoints.csv
    output/kitagawa_templated_composition.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (CRIT, W_BASE, W_EVENT, W_PRE_TRANSITION, bh, col,
                    event_frame, win, write)


CONSENSUS_WEEKS = [
    "2024-11-18",
    "2024-12-02",
    "2025-01-06",
    "2025-02-24",
    "2025-04-28",
    "2025-05-12",
    "2025-07-21",
]

# Five non-Nov/Dec consensus weeks (model-mix Kitagawa already covered
# Nov 18 / Dec 2 in the prior WildChat paper).
FIVE_CHANGEPOINTS = [
    "2025-01-06",
    "2025-02-24",
    "2025-04-28",
    "2025-05-12",
    "2025-07-21",
]


def kitagawa(pre: pd.DataFrame, post: pd.DataFrame, column: str) -> dict:
    """Kitagawa split + per-group contributions across templated/ordinary."""
    mix = within = var_w = 0.0
    parts = {}
    for g, label in ((False, "ordinary"), (True, "templated")):
        p1 = pre.loc[pre.templated == g, column].dropna()
        p2 = post.loc[post.templated == g, column].dropna()
        n1, n2 = len(pre), len(post)
        s1 = len(p1) / n1 if n1 else 0.0
        s2 = len(p2) / n2 if n2 else 0.0
        y1 = p1.mean() if len(p1) else np.nan
        y2 = p2.mean() if len(p2) else np.nan
        if np.isnan(y1) and np.isnan(y2):
            parts[label] = dict(
                s1=s1, s2=s2, y1=y1, y2=y2, dy=np.nan,
                mix=0.0, within=0.0, contrib=0.0,
                n1=0, n2=0,
            )
            continue
        ybar = float(np.nanmean([y1, y2]))
        dy = 0.0 if (np.isnan(y1) or np.isnan(y2)) else float(y2 - y1)
        sbar = (s1 + s2) / 2
        mix_part = (s2 - s1) * ybar
        within_part = sbar * dy
        contrib = (s2 * y2 if not np.isnan(y2) else 0.0) - (
            s1 * y1 if not np.isnan(y1) else 0.0)
        mix += mix_part
        within += within_part
        se1 = float(p1.sem()) if len(p1) > 1 else 0.0
        se2 = float(p2.sem()) if len(p2) > 1 else 0.0
        var_w += sbar ** 2 * (se1 ** 2 + se2 ** 2)
        parts[label] = dict(
            s1=s1, s2=s2, y1=y1, y2=y2, dy=dy,
            mix=mix_part, within=within_part, contrib=contrib,
            n1=len(p1), n2=len(p2),
        )

    total = float(post[column].mean() - pre[column].mean())
    z = within / np.sqrt(var_w) if var_w > 0 else np.nan
    within_p = (float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan)

    def share(x):
        return x / total if total and np.isfinite(total) else np.nan

    T, O = parts["templated"], parts["ordinary"]
    return dict(
        total=total,
        mix=mix,
        within=within,
        residual=total - mix - within,
        mix_share=share(mix),
        within_share=share(within),
        within_p=within_p,
        # Per-group Kitagawa parts
        mix_templated=T["mix"],
        mix_ordinary=O["mix"],
        within_templated=T["within"],
        within_ordinary=O["within"],
        within_templated_share=share(T["within"]),
        within_ordinary_share=share(O["within"]),
        # Algebraic contribution s2*y2 - s1*y1 (= mix_g + within_g)
        contrib_templated=T["contrib"],
        contrib_ordinary=O["contrib"],
        contrib_templated_share=share(T["contrib"]),
        contrib_ordinary_share=share(O["contrib"]),
        # Levels / shares
        s_templated_pre=T["s1"],
        s_templated_post=T["s2"],
        y_templated_pre=T["y1"],
        y_templated_post=T["y2"],
        y_ordinary_pre=O["y1"],
        y_ordinary_post=O["y2"],
        n_pre=len(pre),
        n_post=len(post),
        n_templated_pre=T["n1"],
        n_templated_post=T["n2"],
        n_ordinary_pre=O["n1"],
        n_ordinary_post=O["n2"],
    )


def decompose_pair(pre, post, comparison: str) -> pd.DataFrame:
    rows = []
    for name in CRIT:
        r = kitagawa(pre, post, col(name))
        rows.append({"comparison": comparison, "criterion": name, **r})
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame, title: str) -> None:
    cols = [
        "criterion", "total", "mix", "within",
        "mix_share", "within_templated_share", "within_ordinary_share",
        "contrib_templated_share",
        "s_templated_pre", "s_templated_post",
        "y_templated_pre", "y_templated_post",
        "y_ordinary_pre", "y_ordinary_post",
    ]
    print(f"\n=== {title} ===")
    print(df[cols].round(3).to_string(index=False))


def main():
    E = event_frame()

    pairs = [
        ("Apr15-23 vs Apr25-27 (event)",
         win(E, W_BASE), win(E, W_EVENT)),
        ("Apr8-14 vs Apr15-23 (transition)",
         win(E, W_PRE_TRANSITION), win(E, W_BASE)),
    ]
    W = pd.concat([decompose_pair(pre, post, label)
                   for label, pre, post in pairs], ignore_index=True)
    W["within_p_bh"] = bh(W["within_p"].values)
    write(W.round(6), "kitagawa_templated_windows.csv")

    cp_rows = []
    for wk in CONSENSUS_WEEKS:
        bd = pd.Timestamp(wk)
        win_ = E[(E.date >= bd - pd.Timedelta(days=14))
                 & (E.date < bd + pd.Timedelta(days=14))]
        pre, post = win_[win_.date < bd], win_[win_.date >= bd]
        block = decompose_pair(pre, post, f"CP {wk} (+/-14d)")
        block["week_end"] = wk
        block["in_five"] = wk in FIVE_CHANGEPOINTS
        cp_rows.append(block)
    CP = pd.concat(cp_rows, ignore_index=True)
    CP["within_p_bh"] = bh(CP["within_p"].values)
    write(CP.round(6), "kitagawa_templated_changepoints.csv")

    comp = []
    for label, pre, post in pairs:
        for arm, d in (("pre", pre), ("post", post)):
            comp.append({
                "comparison": label, "arm": arm, "n": len(d),
                "templated_pct": 100 * d.templated.mean(),
                "templated_n": int(d.templated.sum()),
                "ordinary_n": int((~d.templated).sum()),
            })
    for wk in CONSENSUS_WEEKS:
        bd = pd.Timestamp(wk)
        win_ = E[(E.date >= bd - pd.Timedelta(days=14))
                 & (E.date < bd + pd.Timedelta(days=14))]
        pre, post = win_[win_.date < bd], win_[win_.date >= bd]
        for arm, d in (("pre", pre), ("post", post)):
            comp.append({
                "comparison": f"CP {wk} (+/-14d)", "arm": arm, "n": len(d),
                "templated_pct": 100 * d.templated.mean(),
                "templated_n": int(d.templated.sum()),
                "ordinary_n": int((~d.templated).sum()),
            })
    write(pd.DataFrame(comp).round(2), "kitagawa_templated_composition.csv")

    # Headlines
    ev = W[W.comparison.str.startswith("Apr15-23")]
    fmt_table(ev, "Apr 15-23 vs Apr 25-27 (event)")
    v = ev[ev.criterion == "Validation"].iloc[0]
    print(
        f"\n  VALIDATION headline:\n"
        f"    pooled Δμ = {v.total:+.3f}\n"
        f"    mix (share change) = {v.mix:+.3f} ({100*v.mix_share:.1f}% of jump)\n"
        f"    within (behavior)  = {v.within:+.3f} ({100*v.within_share:.1f}%)\n"
        f"      of which templated within = {v.within_templated:+.3f} "
        f"({100*v.within_templated_share:.1f}% of jump)\n"
        f"      of which ordinary within  = {v.within_ordinary:+.3f} "
        f"({100*v.within_ordinary_share:.1f}% of jump)\n"
        f"    templated share {100*v.s_templated_pre:.1f}% -> "
        f"{100*v.s_templated_post:.1f}%\n"
        f"    templated mean  {v.y_templated_pre:.2f} -> {v.y_templated_post:.2f}\n"
        f"    ordinary mean   {v.y_ordinary_pre:.2f} -> {v.y_ordinary_post:.2f}"
    )

    tr = W[W.comparison.str.startswith("Apr8-14")]
    fmt_table(tr, "Apr 8-14 vs Apr 15-23 (transition)")

    print("\n=== Five non-Nov/Dec consensus weeks (Validation) ===")
    five = CP[(CP.in_five) & (CP.criterion == "Validation")]
    print(five[[
        "week_end", "total", "mix", "within", "mix_share",
        "within_templated_share", "within_ordinary_share",
        "s_templated_pre", "s_templated_post",
    ]].round(3).to_string(index=False))

    print("\n=== All consensus weeks x all criteria: mix_share / within_templated_share ===")
    for wk in CONSENSUS_WEEKS:
        block = CP[CP.week_end == wk]
        print(f"\n-- CP {wk} --")
        print(block[[
            "criterion", "total", "mix", "within", "mix_share",
            "within_templated_share", "within_ordinary_share",
        ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
