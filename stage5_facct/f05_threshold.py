"""How much does the April event result depend on the template threshold?
The package flags an opener shared by >= 100 conversations WITHIN the event sample
(~38k conversations). The ordinary draws excluded openers shared by >= 100 across the
much larger pool they were sampled from. Sweep the within-sample threshold down."""
from fcommon import *

def main():
    E = event_scored()
    rows, cons = [], []
    for t in [100, 50, 20, 10, 5, 3]:
        tmpl = E.template_conv_n >= t
        O = E[~tmpl]
        shares = {w: 100 * tmpl[(E.date >= W[0]) & (E.date < W[1])].mean() for w, W in [("base", W_BASE), ("event", W_EVENT), ("post", W_POST)]}
        ps = []
        for c in CRIT:
            r = clustered_contrast(win(O, W_BASE), win(O, W_EVENT), col(c))
            ps.append(r["p"]); rows.append(dict(threshold=t, criterion=c, d=r["d"], lo=r["lo"], hi=r["hi"], p=r["p"],
                                               n_base=r["n_base"], n_event=r["n_event"], **{f"templated_pct_{k}": v for k, v in shares.items()},
                                               templated_pct_all=100 * tmpl.mean()))
        _, cps = changepoints(O); cw = consensus_weeks(cps)
        cons.append(dict(threshold=t, consensus_weeks="; ".join(f"{x.date()}({n})" for x, n in cw.items()),
                         apr28_flagged=any(str(x.date()) == "2025-04-28" for x in cw)))
    R = pd.DataFrame(rows)
    R["p_bh"] = R.groupby("threshold").p.transform(lambda s: bh(s.values))
    writef(R, "f05_threshold_contrasts.csv"); writef(pd.DataFrame(cons), "f05_threshold_consensus.csv")
    v = R[R.criterion == "Validation"][["threshold", "templated_pct_all", "templated_pct_base", "templated_pct_event", "templated_pct_post", "n_base", "n_event", "d", "lo", "hi", "p_bh"]]
    print(v.round(3).to_string())
    print(R.groupby("threshold").apply(lambda g: pd.Series({"n_sig_bh": int((g.p_bh < .05).sum()), "max_abs_d": g.d.abs().max()})).to_string())
    print(pd.DataFrame(cons).to_string())

if __name__ == "__main__":
    main()
