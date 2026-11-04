"""Ordinary draws: era contrast, replication, event window, post-rollback slope,
segmented model, within-post composition, sensitivity to empty openers."""
from fcommon import *
from scipy import stats

def lang_dummies(d, top=8):
    keep = d.language.value_counts().index[:top]
    return pd.get_dummies(d.language.where(d.language.isin(keep), "Other"), prefix="L", drop_first=True, dtype=float)

def main():
    D = {k: draw(k).assign(date=lambda x: x.ts.dt.normalize()) for k in "AB"}
    rows = []
    for k, d in D.items():
        pre, post = win(d, PRE), win(d, POST)
        r = points_contrast(pre, post); r.update(draw=k, spec="era, raw"); rows.append(r)
        r = points_contrast(pre, post, covars=["la"]); r.update(draw=k, spec="era, log assistant words"); rows.append(r)
        r = points_contrast(pre, post, covars=["la", "lu"]); r.update(draw=k, spec="era, log user+assistant words"); rows.append(r)
        # single-model baseline: Mar 1-Apr 14 is entirely gpt-4o-mini
        r = points_contrast(win(d, ("2025-03-01", "2025-04-15")), post); r.update(draw=k, spec="era, Mar 1-Apr 14 baseline"); rows.append(r)
        # drop conversations with an empty opening user message
        empty = set(d[(d.assistant_turn_index == 1) & (d.uw == 0)].conv_id)
        dd = d[~d.conv_id.isin(empty)]
        r = points_contrast(win(dd, PRE), win(dd, POST)); r.update(draw=k, spec="era, empty openers dropped"); rows.append(r)
        # disclosed event window: Apr 15-23 vs Apr 25-27
        b, e = win(d, W_BASE), win(d, W_EVENT)
        r = points_contrast(b, e); r.update(draw=k, spec="event window (Apr25-27 vs Apr15-23)",
                                           model_share_base=(b.model == "gpt-4.1-mini-2025-04-14").mean(),
                                           model_share_event=(e.model == "gpt-4.1-mini-2025-04-14").mean()); rows.append(r)
        # the transition itself: Apr 8-14 vs Apr 15-23
        r = points_contrast(win(d, W_PRE_TRANSITION), b); r.update(draw=k, spec="transition (Apr15-23 vs Apr8-14)"); rows.append(r)
    R = pd.DataFrame(rows); writef(R, "f02_contrasts.csv"); print(R[["draw","spec","n_a","n_b","mean_a","mean_b","diff","lo","hi","p","d"]].round(3).to_string())

    # ---- replication of the weekly series
    wA, wB = weekly(D["A"]), weekly(D["B"])
    j = wA[["mean"]].join(wB[["mean"]], lsuffix="_A", rsuffix="_B", how="inner")
    r, _ = stats.pearsonr(j.mean_A, j.mean_B); rho, _ = stats.spearmanr(j.mean_A, j.mean_B)
    rep = pd.DataFrame([dict(weeks=len(j), pearson_r=r, spearman_rho=rho, mean_abs_diff=(j.mean_A - j.mean_B).abs().mean())])
    writef(rep, "f02_replication.csv"); print(rep.round(3).to_string())
    j.reset_index().to_csv(os.path.join(OUT_F, "f02_weekly_AB.csv"), index=False)

    # ---- post-rollback slope under one frozen identifier (May 1 - Jul 31)
    rows = []
    for k, d in D.items():
        p = win(d, POST_ROLLBACK).copy()
        assert (p.model == "gpt-4.1-mini-2025-04-14").all()
        p["wk"] = (p.ts - pd.Timestamp(POST_ROLLBACK[0])).dt.total_seconds() / (7 * 86400)
        for spec, X in [("raw", p[["wk"]]),
                        ("+ log assistant words", p[["wk", "la"]]),
                        ("+ log assistant words + language", pd.concat([p[["wk", "la"]], lang_dummies(p)], axis=1)),
                        ("+ log user & assistant words + language + turn index", pd.concat([p[["wk", "la", "lu"]].assign(ti=np.log1p(p.assistant_turn_index)), lang_dummies(p)], axis=1))]:
            m = ols_cluster(p.validation, X, p[CLUSTER_UNIT])
            rows.append(dict(draw=k, spec=spec, n=len(p), weeks=p.wk.max(), slope=m.params[1], lo=m.params[1] - 1.96 * m.bse[1],
                             hi=m.params[1] + 1.96 * m.bse[1], p=m.pvalues[1]))
    S = pd.DataFrame(rows); writef(S, "f02_postrollback_slope.csv"); print(S.round(4).to_string())

    # ---- segmented model over the whole span (supplement)
    rows = []
    for k, d in D.items():
        x = win(d, SPAN).copy()
        x["t"] = (x.ts - pd.Timestamp(SPAN[0])).dt.total_seconds() / (7 * 86400)
        x["post"] = (x.ts >= TRANSITION).astype(float)
        x["t_post"] = ((x.ts - TRANSITION).dt.total_seconds() / (7 * 86400)).clip(lower=0) * x.post
        for spec, cols in [("raw", ["t", "post", "t_post"]), ("+ log assistant words + language", None)]:
            X = x[["t", "post", "t_post"]] if cols else pd.concat([x[["t", "post", "t_post", "la"]], lang_dummies(x)], axis=1)
            m = ols_cluster(x.validation, X, x[CLUSTER_UNIT])
            for i, nm in enumerate(["pre_slope", "level_shift", "post_slope_change"], start=1):
                rows.append(dict(draw=k, spec=spec, term=nm, est=m.params[i], lo=m.params[i] - 1.96 * m.bse[i], hi=m.params[i] + 1.96 * m.bse[i], p=m.pvalues[i]))
    G = pd.DataFrame(rows); writef(G, "f02_segmented.csv"); print(G.round(4).to_string())

    # ---- composition inside the post-rollback period: first vs last three complete weeks
    rows = []
    for k, d in D.items():
        p = win(d, POST_ROLLBACK)
        g = p.groupby(pd.Grouper(key="ts", freq=PELT_RESAMPLE))
        wk = pd.DataFrame({"validation": g.validation.mean(), "turns": g.size(),
                           "turns_per_conv": g.apply(lambda z: z.groupby("conv_id").turn_count.first().mean()),
                           "english_pct_turns": g.apply(lambda z: 100 * (z.language == "English").mean()),
                           "median_aw": g.aw.median()})
        wk = wk[wk.turns > 0]
        wk = wk[wk.index <= "2025-07-28"]            # drop the partial Jul 29-31 bin
        for c in ["validation", "turns_per_conv", "english_pct_turns", "median_aw"]:
            rho, pv = stats.spearmanr(np.arange(len(wk)), wk[c])
            rows.append(dict(draw=k, measure=c, weeks=len(wk), first3=wk[c].iloc[:3].mean(), last3=wk[c].iloc[-3:].mean(), spearman_rho=rho, p=pv))
    P = pd.DataFrame(rows); writef(P, "f02_postrollback_composition.csv"); print(P.round(3).to_string())

if __name__ == "__main__":
    main()
