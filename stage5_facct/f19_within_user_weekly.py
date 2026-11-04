"""f19 - two checks on the ordinary-draw result (27B validation).

1. Returning users. A hashed IP stands in for a user. Keep IPs seen both before April 15
   (PRE) and after the rollback (POST) and compare each IP only with itself:
     - fixed-effect regression (one intercept per IP), every turn counted, SE clustered on IP
     - same, adjusted for log reply length, language and turn index
     - same, without the five IPs with the most turns
     - same, English-language turns only
     - each IP weighted equally: mean of per-IP (post - pre) differences, t interval,
       share of IPs that score higher, Wilcoxon signed-rank p
   plus mean validation of leavers, stayers (before and after) and joiners.
2. Post-rollback trend on weekly means. The 12 complete W-MON bins from May 5 to July 28,
   slope per calendar week (the empty bin ending June 2 is a gap), Newey-West SE with 2 and
   4 lags; all turns and English-language turns.
Writes output/facct/f19_within_user.csv, f19_within_user_groups.csv, f19_weekly_hac.csv.
"""
from fcommon import *
import statsmodels.formula.api as smf
from scipy import stats

def fe(B, extra=""):
    m = smf.ols("validation ~ e + C(sender_id)" + extra, data=B).fit(cov_type="cluster", cov_kwds={"groups": B.sender_id})
    return m.params["e"], m.params["e"] - 1.96 * m.bse["e"], m.params["e"] + 1.96 * m.bse["e"], m.pvalues["e"]

def main():
    rows, groups, hac = [], [], []
    for k in "AB":
        d = draw(k)
        pre, post = win(d, PRE), win(d, POST)
        both = set(pre.sender_id) & set(post.sender_id)
        B = pd.concat([pre.assign(e=0.0), post.assign(e=1.0)])
        B = B[B.sender_id.isin(both)].copy(); B["la"] = np.log1p(B.aw)
        tpi = B.groupby("sender_id").size()
        base = dict(draw=k, ips_pre=pre.sender_id.nunique(), ips_post=post.sender_id.nunique(), ips_both=len(both),
                    turns_both=len(B), pct_post_turns_from_returning=100 * post.sender_id.isin(both).mean(),
                    median_turns_per_ip=tpi.median(), top5_share_pct=100 * tpi.nlargest(5).sum() / len(B))
        def add(spec, est, lo, hi, p, n_ips): rows.append(dict(base, spec=spec, est=est, lo=lo, hi=hi, p=p, n_ips=n_ips))
        add("fixed effect, turns", *fe(B), len(both))
        add("fixed effect, + log reply length + language + turn index", *fe(B, " + la + C(language) + assistant_turn_index"), len(both))
        top = tpi.nlargest(5).index; B5 = B[~B.sender_id.isin(top)]
        add("fixed effect, without five heaviest IPs", *fe(B5), len(both) - 5)
        Be = B[B.language == "English"]; bothE = set(Be[Be.e == 0].sender_id) & set(Be[Be.e == 1].sender_id); Be = Be[Be.sender_id.isin(bothE)]
        add("fixed effect, English-language turns", *fe(Be), len(bothE))
        g = B.groupby(["sender_id", "e"]).validation.mean().unstack(); diff = g[1.0] - g[0.0]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(dict(base, spec="each IP weighted equally", est=diff.mean(), lo=diff.mean() - 1.96 * se, hi=diff.mean() + 1.96 * se,
                         p=stats.ttest_1samp(diff, 0).pvalue, n_ips=len(diff), pct_ips_higher=100 * (diff > 0).mean(),
                         wilcoxon_p=stats.wilcoxon(diff).pvalue))
        groups += [dict(draw=k, group="leavers (before)", mean=pre[~pre.sender_id.isin(both)].validation.mean()),
                   dict(draw=k, group="returning (before)", mean=pre[pre.sender_id.isin(both)].validation.mean()),
                   dict(draw=k, group="returning (after)", mean=post[post.sender_id.isin(both)].validation.mean()),
                   dict(draw=k, group="joiners (after)", mean=post[~post.sender_id.isin(both)].validation.mean())]
        d["wkbin"] = d.ts.dt.to_period("W-MON").dt.end_time.dt.normalize()
        for lab, dd in [("all", d), ("English", d[d.language == "English"])]:
            W = dd.groupby("wkbin").validation.agg(["mean", "size"]).reset_index()
            P = W[(W.wkbin >= pd.Timestamp("2025-05-05")) & (W.wkbin <= pd.Timestamp("2025-07-28"))].copy()
            P["t"] = (P.wkbin - P.wkbin.min()).dt.days / 7
            for lag in (2, 4):
                m = sm.OLS(P["mean"].values, sm.add_constant(P[["t"]].values)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
                hac.append(dict(draw=k, turns=lab, weeks=len(P), lag=lag, slope=m.params[1], lo=m.params[1] - 1.96 * m.bse[1],
                                hi=m.params[1] + 1.96 * m.bse[1], p=m.pvalues[1]))
    writef(pd.DataFrame(rows), "f19_within_user.csv"); writef(pd.DataFrame(groups), "f19_within_user_groups.csv")
    writef(pd.DataFrame(hac), "f19_weekly_hac.csv")
    print(pd.DataFrame(rows)[["draw", "spec", "n_ips", "est", "lo", "hi", "p"]].round(3).to_string())
    print(pd.DataFrame(hac).round(3).to_string())

if __name__ == "__main__":
    main()
