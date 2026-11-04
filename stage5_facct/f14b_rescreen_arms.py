"""Table 2 ordinary column under pool-level (full WildChat) template counts, plus sensitivity arms."""
from fcommon import *
GREET = None
def main():
    E = event_scored(); E["date"] = E.ts.dt.normalize()
    k = pd.read_csv(os.path.join(DATA, "greeting_conv.csv"))  # conv_id of pure-greeting openers (stage1_frames/build_poolscreen.py)
    g = E.conv_id.isin(set(k.conv_id))
    arms = {"pool <100 (main)": E[E.template_conv_n < 100],
            "pool <20": E[E.template_conv_n < 20],
            "pool <5": E[E.template_conv_n < 5],
            "pool <100, short greetings kept": E[(E.template_conv_n < 100) | g],
            "within-sample <100 (v3)": E[E.template_conv_n_within < 100]}
    rows = []
    for name, d in arms.items():
        pre, post = win(d, PRE), win(d, POST); pr = win(d, POST_ROLLBACK).copy()
        pr["wk"] = (pr.ts - pd.Timestamp(POST_ROLLBACK[0])).dt.total_seconds() / (7 * 86400)
        for c in CRIT:
            cc = col(c); a_, b_ = pre.dropna(subset=[cc]), post.dropna(subset=[cc])
            r = points_contrast(a_, b_, col=cc); ra = points_contrast(a_, b_, col=cc, covars=["la"])
            _, sp = cohens_d(b_[cc], a_[cc])
            q = pr.dropna(subset=[cc]); m = ols_cluster(q[cc], q[["wk"]], q[CLUSTER_UNIT])
            rows.append(dict(arm=name, criterion=c, n_pre=len(a_), n_post=len(b_), diff=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p"], d=r["d"],
                             diff_lenadj=ra["diff"], d_lenadj=ra["diff"] / sp, p_lenadj=ra["p"],
                             slope=m.params[1], slope_lo=m.params[1]-1.96*m.bse[1], slope_hi=m.params[1]+1.96*m.bse[1], slope_p=m.pvalues[1]))
    R = pd.DataFrame(rows)
    for c in ["p", "p_lenadj", "slope_p"]:
        R[c + "_bh"] = R.groupby("arm")[c].transform(lambda s: bh(s.values))
    writef(R, "f14b_rescreen_arms.csv")
    print(R[["arm","criterion","n_post","diff","d","p_bh","d_lenadj","p_lenadj_bh","slope","slope_p_bh"]].round(3).to_string())
if __name__ == "__main__":
    main()
