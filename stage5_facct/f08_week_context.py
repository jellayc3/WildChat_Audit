"""What changed in the corpus at each consensus week (event frame, package flag)."""
from fcommon import *

def main():
    E = event_scored()
    g = E.groupby(pd.Grouper(key="ts", freq=PELT_RESAMPLE))
    W = pd.DataFrame({"turns": g.size(), "senders": g.sender_id.nunique(), "median_assistant_words": g.aw.median(),
                      "templated_pct": 100 * g.templated.mean(), "top_model": g.model.agg(lambda s: s.value_counts().index[0] if len(s) else None),
                      "top_model_share": g.model.agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)})
    W = W[W.turns > 0]; W["turns_ratio_prev"] = W.turns / W.turns.shift(1)
    W["top_model_changed"] = W.top_model != W.top_model.shift(1)
    writef(W.reset_index().rename(columns={"ts": "week_end"}), "f08_week_context.csv")
    cw = pd.concat([pd.read_csv(os.path.join(OUTPUT, "table5_changepoints_all.csv")).assign(arm="all traffic"),
                    pd.read_csv(os.path.join(OUTPUT, "table8_changepoints_templated_removed.csv")).assign(arm="templated removed")])
    cw["week_end"] = pd.to_datetime(cw.week_end)
    X = cw.merge(W.reset_index().rename(columns={"ts": "week_end"}), on="week_end", how="left")
    writef(X, "f08_consensus_context.csv")
    print(X[["arm", "week_end", "criteria", "turns", "senders", "median_assistant_words", "templated_pct", "top_model", "turns_ratio_prev", "top_model_changed"]].round(2).to_string())
    T = pd.read_csv(os.path.join(OUTPUT, "sec415_per_template_validation.csv"))
    print("templates:", len(T), "validation range", T.validation_mean.min(), T.validation_mean.max())

if __name__ == "__main__":
    main()
