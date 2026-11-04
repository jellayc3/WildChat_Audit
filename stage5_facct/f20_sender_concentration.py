"""f20 - how concentrated each traffic frame is among senders (hashed IPs).
Supporting sample (12B, all traffic and ordinary) and both draws, before (PRE) and after
the rollback (POST): number of IPs, share of turns from the single largest IP and the five
largest, and the templated share of the largest IP's turns. The all-traffic post-rollback
largest IP is the reason all-traffic intervals are wide once errors are clustered on IP.
Writes output/facct/f20_sender_concentration.csv."""
from fcommon import *

def main():
    E = event_scored(); rows = []
    frames = {"supporting, all traffic": E, "supporting, ordinary (<100)": E[E.template_conv_n < 100]}
    for k in "AB": frames[f"Draw {k}"] = draw(k)
    for name, f in frames.items():
        for per, w in [("before", PRE), ("after", POST)]:
            x = win(f, w); vc = x.sender_id.value_counts()
            top = x[x.sender_id == vc.index[0]]
            rows.append(dict(frame=name, period=per, turns=len(x), ips=len(vc), top1_pct=100 * vc.iloc[0] / len(x),
                             top5_pct=100 * vc.iloc[:5].sum() / len(x),
                             top1_templated_pct=100 * top.templated.mean() if "templated" in top else np.nan))
    R = pd.DataFrame(rows); writef(R, "f20_sender_concentration.csv"); print(R.round(1).to_string())

if __name__ == "__main__":
    main()
