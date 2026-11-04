"""12B vs 27B: turn-level agreement on turns scored by both, and weekly-series convergence.
Input data/judge_overlap_turns.csv is a de-identified join (conversation_hash x sender x
assistant_turn_index) of the event sample and the two draws, made from the raw CSVs."""
from fcommon import *
from scipy import stats

def main():
    J = pd.read_csv(os.path.join(DATA, "judge_overlap_turns.csv"))
    J = J.rename(columns={"validation_google_gemma_3_12b_it_score": "v12", "validation_google_gemma_3_27b_it_score": "v27"}).dropna(subset=["v12", "v27"])
    rows = []
    for k, g in list(J.groupby("draw")) + [("A or B (unique turns)", J.drop_duplicates(["wildchat_timestamp", "assistant_turn_index", "v12"]))]:
        r, _ = stats.pearsonr(g.v12, g.v27); rho, _ = stats.spearmanr(g.v12, g.v27)
        rows.append(dict(set=k, turns=len(g), pearson_r=r, spearman_rho=rho, pct_exact=100 * (g.v12 == g.v27).mean(), pct_within_1pt=100 * ((g.v12 - g.v27).abs() <= 1).mean(),
                         mean_12b=g.v12.mean(), mean_27b=g.v27.mean(), mean_diff_27_minus_12=(g.v27 - g.v12).mean()))
    T = pd.DataFrame(rows); writef(T, "f06_judge_turns.csv"); print(T.round(3).to_string())
    # length dependence of disagreement
    g = J[J.draw == "A"].copy(); g["diff"] = g.v27 - g.v12
    g["len_tercile"] = pd.qcut(np.log1p(g.assistant_message_length_words), 3, labels=["short", "mid", "long"])
    L = g.groupby("len_tercile", observed=True).agg(turns=("diff", "size"), mean_diff=("diff", "mean"), r=("v12", lambda s: stats.pearsonr(s, g.loc[s.index, "v27"])[0])).reset_index()
    writef(L, "f06_judge_by_length.csv"); print(L.round(3).to_string())
    # weekly series: event sample with templated removed (12B) vs each draw (27B)
    E = event_scored().rename(columns={col("Validation"): "validation"})
    O = E[~E.templated]; O = O[(O.ts >= SPAN[0]) & (O.ts < SPAN[1])]
    w12 = weekly(O)["mean"]
    rows = []
    for k in "AB":
        d = draw(k); w27 = weekly(d[(d.ts >= SPAN[0]) & (d.ts < SPAN[1])])["mean"]
        j = pd.concat([w12.rename("w12"), w27.rename("w27")], axis=1, join="inner")
        r, p = stats.pearsonr(j.w12, j.w27)
        O2 = O.assign(date=O.ts.dt.normalize()); d2 = d.assign(date=d.ts.dt.normalize())
        s12 = win(O2, POST).validation.mean() - win(O2, PRE).validation.mean()
        s27 = win(d2, POST).validation.mean() - win(d2, PRE).validation.mean()
        rows.append(dict(draw=k, weeks=len(j), pearson_r=r, p=p, shift_12b_templated_removed=s12, shift_27b_draw=s27))
    W = pd.DataFrame(rows); writef(W, "f06_judge_series.csv"); print(W.round(3).to_string())

if __name__ == "__main__":
    main()
