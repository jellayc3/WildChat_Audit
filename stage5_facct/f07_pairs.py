"""Summary of the package's concurrent model-pair analysis (t09) for the paper."""
from fcommon import *

def main():
    P = pd.read_csv(os.path.join(OUTPUT, "table9_pairs_detail.csv"))
    P = P[P.criterion.isin(CORE6)]
    rows = []
    for (pair, tr), g in P.groupby(["pair", "traffic"]):
        rows.append(dict(pair=pair, traffic=tr, overlap_start=g.overlap_start.iloc[0], overlap_end=g.overlap_end.iloc[0],
                         n_a=int(g.n_a.min()), n_b=int(g.n_b.min()),
                         criteria_sep_bh=int((g.p_bh < .05).sum()), criteria_sep_bh_and_perm=int(((g.p_bh < .05) & (g.perm_p_bh < .05)).sum()),
                         max_abs_d=g.d.abs().max()))
    S = pd.DataFrame(rows).sort_values(["traffic", "overlap_end"]); writef(S, "f07_pairs_summary.csv"); print(S.round(2).to_string())
    a = S[S.traffic == "all"]
    print("pairs:", len(a), " separating on >=1 core criterion (clustered BH):", int((a.criteria_sep_bh > 0).sum()),
          " (clustered+permutation BH):", int((a.criteria_sep_bh_and_perm > 0).sum()), " last overlap ends:", a.overlap_end.max())

if __name__ == "__main__":
    main()
