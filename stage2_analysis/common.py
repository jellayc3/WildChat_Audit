"""Estimators and frame access. Imported by every analysis script.

The spine of the design: the unit of analysis is the turn, but the unit of
independence is the cluster (conversation x sender). Everything here follows
from that.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *          # noqa: F401,F403


# ---------------------------------------------------------------------- frames
def event_frame():
    return pd.read_parquet(EVENT_FRAME)


def trend_frame():
    return pd.read_parquet(TREND_FRAME)


def col(name):
    """Display name -> score column."""
    return CRIT[name] + SUF


def win(df, w):
    """Half-open window [w[0], w[1])."""
    return df[(df.date >= w[0]) & (df.date < w[1])]


def ordinary(df):
    """Templated conversations removed, per Hicke & Tomlinson."""
    return df[~df.templated]


def traffic(df, arm):
    return df if arm == "pooled" else ordinary(df)


def out_path(name):
    return os.path.join(OUTPUT, name)


def write(df, name, **kw):
    p = out_path(name)
    df.to_csv(p, index=kw.pop("index", False), **kw)
    print(f"  -> output/{name}")
    return p


# ------------------------------------------------------------------ estimators
def cohens_d(a, b):
    """d = (mean(a) - mean(b)) / pooled SD, from TURN-level dispersion.

    Invariant to the clustering choice: clustering moves the interval, never
    the point estimate. Returns (d, pooled_sd) so callers can convert a
    regression coefficient into d units.
    """
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return ((a.mean() - b.mean()) / sp, sp) if sp > 0 else (np.nan, np.nan)


def bh(p):
    """Benjamini-Hochberg step-up adjusted p-values."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    qv = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, pv[i] * m / (m - rank))
        qv[i] = prev
    q[ok] = qv
    return q


def clustered_contrast(base, event, column, unit=CLUSTER_UNIT, covars=None):
    """Primary test. OLS of the score on a window indicator with cluster-robust
    (Huber-White CR1) standard errors on `unit`.

    Returns d with a 95% interval in d units, the p-value, and the design
    effect: the ratio of clustered to unclustered sampling variance, so the
    cost of dependence is visible per window.

    `d` is always the raw standardised mean difference. When covariates are
    passed, `d_adj` is the covariate-adjusted effect -- the regression
    coefficient in the same d units, and the value the returned interval is
    actually centred on. With no covariates the two are equal by construction,
    which is worth keeping as a check.
    """
    import statsmodels.api as sm

    d, sp = cohens_d(event[column], base[column])
    keep = ["_e", column, unit] + list(covars or [])
    pool = pd.concat([base.assign(_e=0.0), event.assign(_e=1.0)],
                     ignore_index=True)[keep].dropna()
    if pool._e.nunique() < 2 or len(pool) < 3:
        return dict(d=d, d_adj=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                    deff=np.nan, n_base=len(base), n_event=len(event))

    X = sm.add_constant(pool[["_e"] + list(covars or [])].astype(float).values)
    y = pool[column].astype(float).values
    mc = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pool[unit].values})
    mu = sm.OLS(y, X).fit()
    deff = (mc.bse[1] / mu.bse[1]) ** 2 if mu.bse[1] > 0 else np.nan
    return dict(d=d,
                d_adj=mc.params[1] / sp if sp else np.nan,
                lo=(mc.params[1] - 1.96 * mc.bse[1]) / sp if sp else np.nan,
                hi=(mc.params[1] + 1.96 * mc.bse[1]) / sp if sp else np.nan,
                p=mc.pvalues[1], deff=deff,
                n_base=len(base), n_event=len(event))


def cluster_bootstrap(base, event, column, unit=CLUSTER_UNIT,
                      draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Interval support. Resamples whole clusters with replacement, never turns."""
    rng = np.random.default_rng(seed)
    pool = pd.concat([base.assign(_e=0), event.assign(_e=1)],
                     ignore_index=True)[["_e", column, unit]].dropna()
    _, sp = cohens_d(event[column], base[column])
    if not sp:
        return (np.nan, np.nan)
    groups = {k: v for k, v in pool.groupby(unit)}
    keys = np.array(list(groups))
    diffs = []
    for _ in range(draws):
        pick = pd.concat([groups[k] for k in rng.choice(keys, len(keys))],
                         ignore_index=True)
        e, b = pick[pick._e == 1][column], pick[pick._e == 0][column]
        if len(e) and len(b):
            diffs.append((e.mean() - b.mean()) / sp)
    if not diffs:
        return (np.nan, np.nan)
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def cluster_permutation(base, event, column, unit=CLUSTER_UNIT,
                        draws=PERM_DRAWS, seed=PERM_SEED):
    """The distribution-free leg. Shuffles each cluster's window label, keeping
    clusters whole, so the test respects dependence as well as making no
    distributional assumption."""
    rng = np.random.default_rng(seed)
    pool = pd.concat([base.assign(_e=0), event.assign(_e=1)],
                     ignore_index=True)[["_e", column, unit]].dropna()
    if pool.empty or pool._e.nunique() < 2:
        return np.nan
    lab = pool.groupby(unit)._e.first()
    obs = pool[pool._e == 1][column].mean() - pool[pool._e == 0][column].mean()
    vals = pool[column].values
    idx = pd.Series(range(len(lab)), index=lab.index).reindex(pool[unit].values).values
    arr = lab.values.copy()
    hits = 0
    for _ in range(draws):
        rng.shuffle(arr)
        m = arr[idx].astype(bool)
        if m.all() or (~m).all():
            continue
        if abs(vals[m].mean() - vals[~m].mean()) >= abs(obs):
            hits += 1
    return (hits + 1) / (draws + 1)


def icc(df, column, unit=CLUSTER_UNIT):
    """One-way random-effects ICC(1) on the clustering unit."""
    d = df[[column, unit]].dropna()
    if d.empty:
        return np.nan
    g = d.groupby(unit)[column]
    k = g.size()
    if len(k) < 2:
        return np.nan
    msb = g.mean().sub(d[column].mean()).pow(2).mul(k).sum() / (len(k) - 1)
    msw = g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / max(len(d) - len(k), 1)
    k0 = (k.sum() - (k ** 2).sum() / k.sum()) / (len(k) - 1)
    v = (msb - msw) / k0
    return float(max(0.0, v / (v + msw))) if (v + msw) > 0 else np.nan


# --------------------------------------------------------------- changepoints
def weekly_matrix(df, criteria=None):
    """The weeks x criteria matrix the detector runs on.

    Bins are resampled W-MON and LABELLED BY THEIR END DATE, and every non-empty
    week is a bin. The bin labelled 2025-04-28 therefore covers April 22-27:
    the pre-announcement climb plus the whole event window.
    """
    names = criteria or CORE6
    B = df.set_index("ts")
    M = pd.DataFrame({k: B[col(k)].resample(PELT_RESAMPLE).mean() for k in names})
    cnt = B[col(names[0])].resample(PELT_RESAMPLE).count()
    return M[cnt > WEEK_MIN_TURNS].dropna()


def weekly_means(df, column):
    """Single-criterion weekly series on the same bins, with n and sd."""
    g = df.set_index("ts")[column].resample(PELT_RESAMPLE).agg(["mean", "size", "std"])
    return g[g["size"] > WEEK_MIN_TURNS]


def pelt(series, min_size=PELT_MIN_SIZE):
    """PELT (Killick et al. 2012), l2 cost, BIC-style penalty

        pen = log(n) * var(diff(y)) / 2

    Returns positional indices, each the first bin of a new segment.
    """
    import ruptures as rpt
    a = np.asarray(series, float).reshape(-1, 1)
    if len(a) < 2 * min_size + 1:
        return []
    pen = float(np.log(len(a)) * max(np.var(np.diff(a.ravel())) / 2.0, 1e-6))
    bk = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(a).predict(pen=pen)
    return [int(x) for x in bk if x < len(a)]


def pelt_dates(series):
    return [series.index[i] for i in pelt(series.values)]


def changepoints(df, criteria=None):
    """(weekly matrix, {criterion: [week labels]}) for one traffic treatment."""
    M = weekly_matrix(df, criteria)
    return M, {k: [M.index[i] for i in pelt(M[k].values)] for k in M.columns}


def consensus_weeks(cp_by_crit, k=CONSENSUS_K, criteria=None):
    """Weeks flagged by at least k of the six core criteria."""
    c = Counter()
    for name in (criteria or CORE6):
        for d in cp_by_crit.get(name, []):
            c[d] += 1
    return {d: n for d, n in sorted(c.items()) if n >= k}


def consensus_null(M, draws=NULL_DRAWS, seed=PERM_SEED):
    """Correlation-preserving permutation null for the >=k-of-6 rule.

    ONE random permutation of week order is applied to all six columns at once,
    so each week's six values travel together. This preserves the
    cross-criterion correlation that makes the rule weak (validation and
    positive affirmation sit at rho = 0.84) while destroying the temporal
    structure the detector looks for. Permuting each criterion separately would
    assume the correlation away and understate the false-positive rate.

    Returns per-week clearance rates for >=2, >=3, >=4 of six, and the
    distribution of consensus-week counts per synthetic 37-week series.
    """
    rng = np.random.default_rng(seed)
    n = len(M)
    tally = {2: 0, 3: 0, 4: 0}
    ncons = []
    for _ in range(draws):
        P = M.values[rng.permutation(n)]
        c = Counter()
        for j in range(M.shape[1]):
            for i in pelt(P[:, j]):
                c[i] += 1
        ncons.append(sum(1 for v in c.values() if v >= CONSENSUS_K))
        for v in c.values():
            for k in (2, 3, 4):
                if v >= k:
                    tally[k] += 1
    rates = {k: 100 * v / (draws * n) for k, v in tally.items()}
    return rates, np.array(ncons)


def bootstrap_support(df, cps, draws=BOOT_SUPPORT_DRAWS, seed=SEED):
    """Resample each weekly mean from its standard error, rerun the detector,
    and report the share of resamples recovering a break within two bins."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, dates in cps.items():
        if not dates:
            continue
        g = weekly_means(df, col(name))
        mu = g["mean"].values
        se = np.nan_to_num((g["std"] / np.sqrt(g["size"])).values)
        hits = {d: 0 for d in dates}
        for _ in range(draws):
            got = pelt_dates(pd.Series(mu + rng.normal(0, se), index=g.index))
            for d in hits:
                if any(abs((d - x).days) <= 14 for x in got):
                    hits[d] += 1
        out[name] = {d: 100 * v / draws for d, v in hits.items()}
    return out


# -------------------------------------------------------------------- display
def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<.0001" if p < 1e-4 else f"{p:.4f}".lstrip("0")


def fmt_d(r):
    return f"{r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
