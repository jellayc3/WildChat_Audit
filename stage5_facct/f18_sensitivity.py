"""f18 — low-cost sensitivity checks (B1-B3).
Uses only the existing scored ordinary frames (27B validation); no new annotation.

  B1  English-only and reply-length-tercile strata: era contrast + post-rollback slope
  B2  Raw vs directly standardized weekly validation (language group x reply-length tercile,
      reference weights = pooled pre-period composition)
  B3  Weekly-series inference: era contrast and post-rollback slope on weekly means with
      Newey-West (HAC) errors, to account for common weekly shocks and serial dependence

Run:  python f18_sensitivity.py [data dir] [out dir]   (defaults: ../data, ../output/facct)
Estimators mirror fcommon.py: OLS with CR1 errors clustered on sender_id (hashed IP), as in config.CLUSTER_UNIT.
"""
import sys, os
import numpy as np, pandas as pd
import statsmodels.api as sm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_PKG, "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_PKG, "output", "facct")
os.makedirs(OUT, exist_ok=True)
PRE = ("2024-11-01", "2025-04-15"); POST = ("2025-04-29", "2025-08-01"); POSTR = ("2025-05-01", "2025-08-01")
RESAMPLE = "W-MON"

def win(d, w): return d[(d.ts >= w[0]) & (d.ts < w[1])]

def load(k):
    d = pd.read_parquet(os.path.join(DATA, f"ordinary_{k}_frame.parquet"))
    d = d[d.validation.notna()].copy()
    d["la"] = np.log1p(d.aw)
    d["lang"] = d.language.where(d.language.isin(["English", "Russian", "Chinese"]), "Other")
    return d

def ols_cluster(y, X, g):
    X = sm.add_constant(np.asarray(X, float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})

def contrast(a, b, w=None):
    pool = pd.concat([a.assign(_e=0.0), b.assign(_e=1.0)], ignore_index=True)
    X = sm.add_constant(pool[["_e"]].values.astype(float), has_constant="add")
    if w is None:
        m = sm.OLS(pool.validation.values, X).fit(cov_type="cluster", cov_kwds={"groups": pool.sender_id.values})
    else:
        m = sm.WLS(pool.validation.values, X, weights=pool[w].values).fit(cov_type="cluster", cov_kwds={"groups": pool.sender_id.values})
    return m.params[1], m.params[1] - 1.96 * m.bse[1], m.params[1] + 1.96 * m.bse[1], m.pvalues[1]

def slope(p):
    p = p.copy(); p["wk"] = (p.ts - pd.Timestamp(POSTR[0])).dt.total_seconds() / (7 * 86400)
    m = ols_cluster(p.validation, p[["wk"]], p.sender_id)
    return m.params[1], m.params[1] - 1.96 * m.bse[1], m.params[1] + 1.96 * m.bse[1], m.pvalues[1]

rows, weekly_rows, hac_rows = [], [], []
for k in "AB":
    d = load(k)
    # reproduction check against f02_contrasts ("era, raw") and f02_postrollback_slope ("raw")
    est = contrast(win(d, PRE), win(d, POST)); rows.append(dict(draw=k, check="B0 reproduce", stratum="all", kind="era", n=len(win(d, PRE)) + len(win(d, POST)), est=est[0], lo=est[1], hi=est[2], p=est[3]))
    s = slope(win(d, POSTR)); rows.append(dict(draw=k, check="B0 reproduce", stratum="all", kind="post slope", n=len(win(d, POSTR)), est=s[0], lo=s[1], hi=s[2], p=s[3]))

    # ---- B1 strata
    cuts = np.quantile(win(d, PRE).aw, [1/3, 2/3])          # terciles fixed on the pre period
    d["lenT"] = pd.cut(d.aw, [-1, cuts[0], cuts[1], np.inf], labels=["short", "medium", "long"]).astype(str)
    strata = {"English only": d.language == "English", "Non-English": d.language != "English"}
    for t in ["short", "medium", "long"]:
        strata[f"Reply length tercile: {t}"] = d["lenT"] == t
    for name, mask in strata.items():
        x = d[mask]
        e = contrast(win(x, PRE), win(x, POST)); rows.append(dict(draw=k, check="B1 strata", stratum=name, kind="era", n=len(win(x, PRE)) + len(win(x, POST)), est=e[0], lo=e[1], hi=e[2], p=e[3]))
        s = slope(win(x, POSTR)); rows.append(dict(draw=k, check="B1 strata", stratum=name, kind="post slope", n=len(win(x, POSTR)), est=s[0], lo=s[1], hi=s[2], p=s[3]))

    # ---- B2 direct standardization: cell = lang group x length tercile, reference = pre-period shares
    d["cell"] = d.lang + "|" + d["lenT"]
    ref = win(d, PRE).cell.value_counts(normalize=True)
    d["wkbin"] = d.ts.dt.to_period("W-MON").dt.end_time.dt.normalize()
    share = d.groupby("wkbin").cell.value_counts(normalize=True).rename("obs").reset_index()
    share["ref"] = share.cell.map(ref).fillna(0.0)
    share["w"] = share.ref / share.obs
    d = d.merge(share[["wkbin", "cell", "w"]], on=["wkbin", "cell"], how="left")
    for wk, g in d.groupby("wkbin"):
        cover = ref[ref.index.isin(g.cell.unique())].sum()  # reference mass present this week
        weekly_rows.append(dict(draw=k, week=wk.date(), turns=len(g), raw=g.validation.mean(),
                                standardized=np.average(g.validation, weights=g.w) / 1.0,
                                english_pct=100 * (g.language == "English").mean(), ref_mass_present=cover))
    e = contrast(win(d, PRE), win(d, POST), w="w"); rows.append(dict(draw=k, check="B2 standardized", stratum="lang x length tercile, pre-period weights", kind="era", n=len(win(d, PRE)) + len(win(d, POST)), est=e[0], lo=e[1], hi=e[2], p=e[3]))
    p = win(d, POSTR).copy(); p["wk"] = (p.ts - pd.Timestamp(POSTR[0])).dt.total_seconds() / (7 * 86400)
    X = sm.add_constant(p[["wk"]].values, has_constant="add")
    m = sm.WLS(p.validation.values, X, weights=p.w.values).fit(cov_type="cluster", cov_kwds={"groups": p.sender_id.values})
    rows.append(dict(draw=k, check="B2 standardized", stratum="lang x length tercile, pre-period weights", kind="post slope", n=len(p), est=m.params[1], lo=m.params[1] - 1.96 * m.bse[1], hi=m.params[1] + 1.96 * m.bse[1], p=m.pvalues[1]))

    # ---- B3 weekly-series HAC
    W = d.groupby("wkbin").agg(v=("validation", "mean"), n=("validation", "size")).reset_index()
    W["post"] = (W.wkbin >= pd.Timestamp("2025-05-05")).astype(float)       # bins ending May 5 onward = Apr 29+
    W["pre"] = (W.wkbin <= pd.Timestamp("2025-04-14")).astype(float)
    E = W[(W.pre == 1) | (W.post == 1)]
    for lag in (2, 4):
        m = sm.OLS(E.v.values, sm.add_constant(E[["post"]].values)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        hac_rows.append(dict(draw=k, kind="era (weekly means)", weeks=len(E), lag=lag, est=m.params[1], lo=m.params[1] - 1.96 * m.bse[1], hi=m.params[1] + 1.96 * m.bse[1], p=m.pvalues[1]))
    P = W[(W.wkbin >= pd.Timestamp("2025-05-05")) & (W.wkbin <= pd.Timestamp("2025-07-28"))].copy()  # full post-rollback weeks
    P["t"] = (P.wkbin - P.wkbin.min()).dt.days / 7   # calendar weeks; the empty bin ending Jun 2 is a gap, not a step
    for lag in (2, 4):
        m = sm.OLS(P.v.values, sm.add_constant(P[["t"]].values)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        hac_rows.append(dict(draw=k, kind="post slope (weekly means)", weeks=len(P), lag=lag, est=m.params[1], lo=m.params[1] - 1.96 * m.bse[1], hi=m.params[1] + 1.96 * m.bse[1], p=m.pvalues[1]))

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "f18_sensitivity.csv"), index=False)
Wk = pd.DataFrame(weekly_rows); Wk.to_csv(os.path.join(OUT, "f18_weekly_raw_vs_standardized.csv"), index=False)
H = pd.DataFrame(hac_rows); H.to_csv(os.path.join(OUT, "f18_weekly_hac.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string()); print(H.round(4).to_string())
print(Wk.groupby("draw").ref_mass_present.min())

# ---- figure: raw vs standardized
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for a, k in zip(ax, "AB"):
    w = Wk[Wk.draw == k].copy(); w["week"] = pd.to_datetime(w.week)
    full = pd.DataFrame({"week": pd.date_range(w.week.min(), w.week.max(), freq="W-MON")})
    w = full.merge(w, on="week", how="left"); x = w.week          # NaN rows break lines at coverage gaps
    a.plot(x, w.raw, color="#444", lw=1.6, label="Raw")
    a.plot(x, w.standardized, color="#C00000", lw=1.6, ls="--", label="Standardized to pre-period mix")
    a.axvline(pd.Timestamp("2025-04-15"), color="#888", ls=":", lw=1)
    a.set_title(f"Draw {k}", fontsize=10); a.tick_params(labelsize=8)
    import matplotlib.dates as mdates
    a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax[0].set_ylabel("Mean validation (0-10, 27B)"); ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "FigureS3_raw_vs_standardized_27B.png"), dpi=200)
