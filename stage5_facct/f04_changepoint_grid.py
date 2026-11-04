"""One detector (package PELT spec) on validation, four framings x two weekly aggregations."""
from fcommon import *

def main():
    E = event_scored(); v12 = col("Validation")
    E = E.rename(columns={v12: "validation"})
    frames = {"All traffic, event sample (12B)": E, "Templated removed, event sample (12B)": E[~E.templated],
              "Ordinary Draw A (27B)": draw("A"), "Ordinary Draw B (27B)": draw("B")}
    rows = []
    for name, d in frames.items():
        d = d[(d.ts >= SPAN[0]) & (d.ts < SPAN[1])]
        for how in ["turn", "cluster_mean"]:
            g = weekly(d, "validation", how)
            idx = pelt(g["mean"].values); dates = [g.index[i] for i in idx]
            sup = support(g, dates)
            spring = [x for x in dates if x.month in (4, 5)]
            rows.append(dict(framing=name, weekly_mean=how, bins=len(g),
                             breaks="; ".join(f"{x.date()} ({sup[x]:.0f}%)" for x in dates),
                             first_spring=str(spring[0].date()) if spring else "none",
                             first_spring_support=sup[spring[0]] if spring else np.nan))
    R = pd.DataFrame(rows); writef(R, "f04_changepoint_grid.csv"); print(R.to_string())
    fs = sorted(set(R.first_spring) - {"none"})
    print("distinct first-spring dates:", fs, "span days:", (pd.Timestamp(fs[-1]) - pd.Timestamp(fs[0])).days)

if __name__ == "__main__":
    main()
