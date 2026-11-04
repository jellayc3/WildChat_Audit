"""Figure 9 and Section 4.4: how much each available feature explains.

Two kinds of feature, reported on one axis but not the same statistic, which
the figure caption has to say plainly:

  continuous  (reply length, user length, turn index)  signed Pearson r
  categorical (model identity, language, changepoint segment, templated)
              unsigned sqrt(eta^2) from a one-way ANOVA, which cannot be
              negative and is not directionally comparable to an r

Both are on a 0-1 magnitude scale, which is why they can share an axis at all.

    output/sec44_feature_correlations.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import CORE6, changepoints, col, event_frame, write


def eta(df, feature, value, min_group=30):
    """sqrt(eta^2) for a categorical feature: the share of variance lying
    between groups. Groups smaller than min_group are pooled into 'other' so a
    handful of singleton languages cannot inflate it."""
    d = df[[feature, value]].dropna()
    counts = d[feature].value_counts()
    small = counts[counts < min_group].index
    g = d[feature].where(~d[feature].isin(small), "other")
    groups = [v.values for _, v in d[value].groupby(g) if len(v) > 1]
    if len(groups) < 2:
        return np.nan, 0
    grand = d[value].mean()
    ss_b = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
    ss_t = ((d[value] - grand) ** 2).sum()
    return (np.sqrt(ss_b / ss_t) if ss_t > 0 else np.nan), len(groups)


def main():
    E = event_frame().copy()
    E["log_assistant_words"] = np.log1p(E.aw)
    E["log_user_words"] = np.log1p(E.uw)

    # Changepoint segment: which inter-break period a turn falls in, using the
    # all-traffic consensus breaks. A feature only in the sense that the series
    # has a level that shifts; it is not an explanation of anything.
    _, cps = changepoints(E)
    breaks = sorted({d for v in cps.values() for d in v})
    E["segment"] = np.searchsorted(np.array(breaks, dtype="datetime64[ns]"),
                                   E.ts.values.astype("datetime64[ns]"))
    E["segment"] = E.segment.astype(str)
    E["templated_flag"] = E.templated.map({True: "templated", False: "ordinary"})

    CONTINUOUS = ["log_assistant_words", "log_user_words", "assistant_turn_index"]
    CATEGORICAL = ["model", "language", "segment", "templated_flag"]

    rows = []
    for name in CORE6:
        c = col(name)
        for f in CONTINUOUS:
            d = E[[c, f]].dropna()
            r = stats.pearsonr(d[c], d[f])[0] if len(d) > 2 else np.nan
            rows.append({"criterion": name, "feature": f, "kind": "continuous",
                         "statistic": "pearson_r", "value": round(r, 4),
                         "magnitude": round(abs(r), 4), "n_levels": "",
                         "n": len(d)})
        for f in CATEGORICAL:
            v, k = eta(E, f, c)
            rows.append({"criterion": name, "feature": f, "kind": "categorical",
                         "statistic": "sqrt_eta_squared", "value": round(v, 4),
                         "magnitude": round(v, 4), "n_levels": k,
                         "n": int(E[[c, f]].dropna().shape[0])})

    F = pd.DataFrame(rows)
    write(F, "sec44_feature_correlations.csv")

    piv = F.pivot_table(index="feature", columns="criterion", values="magnitude")
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean", ascending=False).round(3)
    write(piv.reset_index(), "sec44_feature_ranking.csv")
    print(piv.to_string())
    print(f"\n  {len(breaks)} consensus breaks define {E.segment.nunique()} segments")


if __name__ == "__main__":
    main()
