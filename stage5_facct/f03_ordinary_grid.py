"""324-specification grid for the ordinary-draw era contrast, both draws.
4 pre windows x 3 post windows x 3 units x 3 length adjustments x 3 template thresholds."""
from fcommon import *
PRE_ARMS = {"Nov 1-Apr 14": ("2024-11-01", "2025-04-15"), "last 12 weeks (Jan 21-Apr 14)": ("2025-01-21", "2025-04-15"),
            "last 6 weeks (Mar 4-Apr 14)": ("2025-03-04", "2025-04-15"), "Mar 1-Apr 14": ("2025-03-01", "2025-04-15")}
POST_ARMS = {"Apr 29-Jul 31": ("2025-04-29", "2025-08-01"), "May 13-Jul 31": ("2025-05-13", "2025-08-01"), "Jun 1-Jul 31": ("2025-06-01", "2025-08-01")}
UNITS = ["turn", "cluster_mean", "cluster_first"]
LENGTHS = {"none": [], "log assistant": ["la"], "log user + assistant": ["lu", "la"]}
THRESH = {"<100 (as drawn)": 100, "<20": 20, "<5": 5}

def fit(a, b, unit, cov):
    pool = pd.concat([a.assign(_e=0.0), b.assign(_e=1.0)], ignore_index=True)
    if unit == "turn":
        m = ols_cluster(pool.validation, pool[["_e"] + cov], pool[CLUSTER_UNIT])
    else:
        g = pool.sort_values("ts").groupby([UNIT_ID, CLUSTER_UNIT, "_e"])
        agg = g[["validation"] + cov].mean() if unit == "cluster_mean" else g[["validation"] + cov].first()
        agg = agg.reset_index()
        m = ols_cluster(agg.validation, agg[["_e"] + cov], agg[CLUSTER_UNIT])
    return m.params[1], m.bse[1], m.pvalues[1]

def main():
    rows = []
    for k in "AB":
        d = draw(k).assign(date=lambda x: x.ts.dt.normalize())
        for tn, t in THRESH.items():
            dt = d[d.template_conv_n_upstream < t]
            for pn, pw in PRE_ARMS.items():
                for qn, qw in POST_ARMS.items():
                    a, b = win(dt, pw), win(dt, qw)
                    for u in UNITS:
                        for ln, cov in LENGTHS.items():
                            est, se, p = fit(a, b, u, cov)
                            rows.append(dict(draw=k, templates=tn, pre=pn, post=qn, unit=u, length=ln, est=est, se=se, p=p, n_pre=len(a), n_post=len(b)))
    G = pd.DataFrame(rows)
    G["p_bh"] = G.groupby("draw").p.transform(lambda s: bh(s.values))
    writef(G, "f03_ordinary_grid.csv")
    S = G.groupby("draw").agg(n=("est", "size"), positive=("est", lambda s: int((s > 0).sum())),
                              sig_bh=("p_bh", lambda s: int((s < .05).sum())), est_min=("est", "min"), est_median=("est", "median"), est_max=("est", "max"))
    writef(S.reset_index(), "f03_ordinary_grid_summary.csv"); print(S.round(3).to_string())

if __name__ == "__main__":
    main()
