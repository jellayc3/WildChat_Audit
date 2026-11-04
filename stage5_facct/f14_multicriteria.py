"""Six criteria + holistic (12B, event sample): era contrast and post-rollback slope on
(a) all traffic, (b) templated removed at the package threshold (>=100 within sample),
(c) templated removed at a stricter threshold (>=3 within sample, close to the draws' pool-level rule).
    output/facct/f14_multicriteria.csv"""
from fcommon import *

def main():
    E = event_scored(); E["date"] = E.ts.dt.normalize()
    frames = {"all traffic": E, "templated removed (>=100)": E[E.template_conv_n < 100], "templated removed (>=3)": E[E.template_conv_n < 3]}
    rows = []
    for name, d in frames.items():
        pre, post = win(d, PRE), win(d, POST); pr = win(d, POST_ROLLBACK).copy()
        pr["wk"] = (pr.ts - pd.Timestamp(POST_ROLLBACK[0])).dt.total_seconds() / (7 * 86400)
        for c in CRIT:
            cc = col(c)
            r = points_contrast(pre.dropna(subset=[cc]), post.dropna(subset=[cc]), col=cc)
            ra = points_contrast(pre.dropna(subset=[cc]), post.dropna(subset=[cc]), col=cc, covars=["la"])
            q = pr.dropna(subset=[cc]); m = ols_cluster(q[cc], q[["wk"]], q[CLUSTER_UNIT])
            rows.append(dict(frame=name, criterion=c, n_pre=r["n_a"], n_post=r["n_b"], templated_pct_pre=100 * pre.templated.mean(), templated_pct_post=100 * post.templated.mean(),
                             mean_pre=r["mean_a"], mean_post=r["mean_b"], diff=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p"], d=r["d"],
                             diff_lenadj=ra["diff"], lo_lenadj=ra["lo"], hi_lenadj=ra["hi"], p_lenadj=ra["p"],
                             slope=m.params[1], slope_lo=m.params[1] - 1.96 * m.bse[1], slope_hi=m.params[1] + 1.96 * m.bse[1], slope_p=m.pvalues[1]))
    R = pd.DataFrame(rows)
    R["p_bh"] = R.groupby("frame").p.transform(lambda s: bh(s.values)); R["slope_p_bh"] = R.groupby("frame").slope_p.transform(lambda s: bh(s.values))
    writef(R, "f14_multicriteria.csv")
    print(R[["frame", "criterion", "mean_pre", "mean_post", "diff", "d", "p_bh", "diff_lenadj", "p_lenadj", "slope", "slope_p_bh"]].round(3).to_string())

if __name__ == "__main__":
    main()
