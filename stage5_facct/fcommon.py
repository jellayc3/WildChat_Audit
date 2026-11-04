"""Shared helpers for the FAccT analyses (stage 5). Reuses stage-2 estimators."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stage2_analysis"))
from common import *                      # noqa  (config + estimators)
import statsmodels.api as sm

OUT_F = os.path.join(OUTPUT, "facct"); os.makedirs(OUT_F, exist_ok=True)
def writef(df, name):
    p = os.path.join(OUT_F, name); df.to_csv(p, index=False); print(f"  -> output/facct/{name}"); return p

# Declared windows for the ordinary draws (half-open).
PRE  = ("2024-11-01", "2025-04-15")   # before the Apr 15 serving change
POST = ("2025-04-29", "2025-08-01")   # after the rollback; corpus is empty Apr 28-30
POST_ROLLBACK = ("2025-05-01", "2025-08-01")
TRANSITION = pd.Timestamp("2025-04-15")
SPAN = ("2024-11-01", "2025-08-01")

def draw(k):
    d = pd.read_parquet(ORDINARY_FRAME[k])
    d = d[d.validation.notna()].copy()
    d["la"] = np.log1p(d.aw); d["lu"] = np.log1p(d.uw)
    return d

def event_scored():
    E = event_frame()
    E["la"] = np.log1p(E.aw); E["lu"] = np.log1p(E.uw)
    return E

def ols_cluster(y, X, groups):
    X = sm.add_constant(np.asarray(X, float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})

def ols_robust(y, X):
    X = sm.add_constant(np.asarray(X, float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="HC1")

def points_contrast(a, b, col="validation", covars=None, unit=CLUSTER_UNIT):
    """b minus a in raw score points, cluster-robust 95% CI, plus Cohen's d."""
    pool = pd.concat([a.assign(_e=0.0), b.assign(_e=1.0)], ignore_index=True)
    cols = ["_e"] + list(covars or [])
    pool = pool.dropna(subset=[col] + cols)
    m = ols_cluster(pool[col], pool[cols], pool[unit])
    d, sp = cohens_d(b[col], a[col])
    return dict(n_a=len(a), n_b=len(b), mean_a=a[col].mean(), mean_b=b[col].mean(),
                diff=m.params[1], lo=m.params[1]-1.96*m.bse[1], hi=m.params[1]+1.96*m.bse[1],
                p=m.pvalues[1], d=d)

def weekly(df, col="validation", how="turn"):
    B = df.set_index("ts")
    if how == "turn":
        g = B[col].resample(PELT_RESAMPLE).agg(["mean", "count", "std"]).rename(columns={"count": "size"})
    else:
        cm = df.groupby([pd.Grouper(key="ts", freq=PELT_RESAMPLE), UNIT_ID])[col].mean().reset_index()
        g = cm.dropna(subset=[col]).groupby("ts")[col].agg(["mean", "count", "std"]).rename(columns={"count": "size"})
    g = g[g["size"] > 0].copy(); g["se"] = g["std"] / np.sqrt(g["size"])
    return g

def support(g, dates, draws=BOOT_SUPPORT_DRAWS, seed=SEED):
    rng = np.random.default_rng(seed); hits = {d: 0 for d in dates}
    mu, se = g["mean"].values, np.nan_to_num(g["se"].values)
    for _ in range(draws):
        got = [g.index[i] for i in pelt(mu + rng.normal(0, se))]
        for d in hits:
            if any(abs((d - x).days) <= 14 for x in got): hits[d] += 1
    return {d: 100 * v / draws for d, v in hits.items()}
