"""Sample accounting, serving concurrency, application share, composition drift."""
from fcommon import *

def main():
    E = event_scored(); T = trend_frame(); A = draw("A"); B = draw("B")
    rawA = pd.read_parquet(ORDINARY_FRAME["A"]); rawB = pd.read_parquet(ORDINARY_FRAME["B"])
    # ---- sample accounting
    rows = []
    for name, d, raw in [("Event sample (12B)", E, E), ("Trend sample (12B)", T, T), ("Ordinary Draw A (27B)", A, rawA), ("Ordinary Draw B (27B)", B, rawB)]:
        wk = d.groupby(pd.Grouper(key="ts", freq=PELT_RESAMPLE)).cluster_id.nunique()
        wk = wk[wk > 0]
        rows.append(dict(sample=name, turns_drawn=len(raw), turns_scored=len(d), clusters=d.cluster_id.nunique(),
                         conversations=d.conv_id.nunique(), senders=d.sender_id.nunique(),
                         first=str(d.ts.min().date()), last=str(d.ts.max().date()),
                         weeks=len(wk), clusters_per_week_median=float(wk.median()), clusters_per_week_min=int(wk.min())))
    S = pd.DataFrame(rows); writef(S, "f01_samples.csv"); print(S.to_string())
    print("total scored turns:", int(S.turns_scored.sum()))
    # shared conversations between draws: not identifiable after opaque ids -> checked upstream (0 shared).

    # ---- serving concurrency, per month, on every frame
    out = []
    for name, d in [("trend", T), ("event", E), ("A", A), ("B", B)]:
        m = d.groupby(["month", "model"]).size().unstack(fill_value=0)
        sh = m.div(m.sum(1), axis=0)
        for mo, r in sh.iterrows():
            out.append(dict(frame=name, month=str(mo.date())[:7], turns=int(m.loc[mo].sum()),
                            labels_any=int((m.loc[mo] > 0).sum()), labels_10pct=int((r >= .10).sum()),
                            top=r.idxmax(), top_share=round(float(r.max()), 4)))
    C = pd.DataFrame(out); writef(C, "f01_concurrency_by_month.csv")
    t = C[C.frame == "trend"]
    print("trend months:", len(t), " months with >=2 labels at >=10%:", int((t.labels_10pct >= 2).sum()),
          " last such month:", t[t.labels_10pct >= 2].month.max())
    # serving-window overlap from the package's own windows (timestamps), o1 excluded
    W = T[~T.model.isin(PAIR_EXCLUDE)].groupby("model").ts.agg(["min", "max"])
    ov = []
    ms = list(W.index)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            s = max(W.loc[ms[i], "min"], W.loc[ms[j], "min"]); e = min(W.loc[ms[i], "max"], W.loc[ms[j], "max"])
            if e > s: ov.append(dict(a=ms[i], b=ms[j], start=s, end=e, days=(e - s).days))
    OV = pd.DataFrame(ov).sort_values("end"); writef(OV, "f01_serving_overlaps.csv")
    print("last date two (non-o1) labels overlap:", OV.end.max(), OV.sort_values("end").tail(2)[["a","b","end"]].to_string())
    # after the transition: single label?
    for name, d in [("trend", T), ("event", E), ("A", A), ("B", B)]:
        after = d[d.ts >= "2025-04-16"]
        print(f"  {name}: share gpt-4.1-mini-2025-04-14 on/after Apr 16 = {(after.model=='gpt-4.1-mini-2025-04-14').mean():.4f}  (n={len(after)})")
        mid = d[(d.ts >= "2025-01-06") & (d.ts < "2025-04-15")]
        print(f"     Jan 6-Apr 14 labels: {mid.model.value_counts().to_dict()}")

    # ---- application (templated) share over time, event frame
    wk = E.set_index("ts").templated.resample(PELT_RESAMPLE).agg(["mean", "size"])
    wk = wk[wk["size"] > 0]; wk["templated_pct"] = 100 * wk["mean"]
    writef(wk.reset_index()[["ts", "size", "templated_pct"]], "f01_weekly_templated_share.csv")
    mo = E.groupby("month").templated.mean() * 100
    writef(mo.reset_index().rename(columns={"templated": "templated_pct"}), "f01_monthly_templated_share.csv")
    print("monthly templated % (event):", mo.round(1).to_dict())
    # thinning of the filtered frame
    O = E[~E.templated]
    wkconv = O.groupby(pd.Grouper(key="ts", freq=PELT_RESAMPLE)).conv_id.nunique(); wkconv = wkconv[wkconv > 0]
    post = wkconv[wkconv.index > "2025-04-15"]
    print("filtered event frame: min weekly conversations after Apr 15:", int(post.min()), "week", post.idxmin().date(),
          "| median before Apr 15:", float(wkconv[wkconv.index <= "2025-04-15"].median()))
    writef(wkconv.rename("conversations").reset_index(), "f01_filtered_weekly_conversations.csv")

    # ---- near-threshold application traffic inside the ordinary draws
    rows = []
    for k, d in [("A", A), ("B", B)]:
        for per, w in [("pre", PRE), ("post", POST)]:
            x = win(d.assign(date=d.ts.dt.normalize()), w)
            rows.append(dict(draw=k, period=per, turns=len(x),
                             pct_tcn_ge20=100 * (x.template_conv_n_upstream >= 20).mean(),
                             pct_tcn_ge5=100 * (x.template_conv_n_upstream >= 5).mean()))
        rows.append(dict(draw=k, period="all", turns=len(d), pct_tcn_ge20=100 * (d.template_conv_n_upstream >= 20).mean(),
                         pct_tcn_ge5=100 * (d.template_conv_n_upstream >= 5).mean()))
    N = pd.DataFrame(rows); writef(N, "f01_near_threshold.csv"); print(N.round(2).to_string())

    # ---- composition drift inside the ordinary draws, pre vs post
    rows = []
    for k, d in [("A", A), ("B", B)]:
        d = d.assign(date=d.ts.dt.normalize())
        for per, w in [("pre", PRE), ("post", POST)]:
            x = win(d, w)
            conv = x.groupby("conv_id").agg(tc=("turn_count", "first"), lang=("language", "first"))
            rows.append(dict(draw=k, period=per, conversations=len(conv),
                             turns_per_conversation=conv.tc.mean(),
                             english_pct_conversations=100 * (conv.lang == "English").mean(),
                             english_pct_turns=100 * (x.language == "English").mean(),
                             median_assistant_words=x.aw.median(),
                             gpt41mini_pct=100 * (x.model == "gpt-4.1-mini-2025-04-14").mean()))
            mix = x.model.value_counts(normalize=True).round(3).to_dict()
            print(k, per, "model mix:", mix)
    D = pd.DataFrame(rows); writef(D, "f01_composition_drift.csv"); print(D.round(2).to_string())
    # empty opening user message (first turn with zero user words)
    for k, d in [("A", A), ("B", B)]:
        f = d[d.assistant_turn_index == 1]
        print(k, "conversations with empty opening message:", int((f.uw == 0).sum()))

if __name__ == "__main__":
    main()
