"""Section 4.1.5: where the April break goes once templated conversations are
removed, and the Apr 15 model-transition contrast.

With templated traffic removed the April 22-27 bin is no longer flagged and the
break moves a week earlier, onto the gpt-4o-mini -> gpt-4.1-mini swap. The
contrast on either side of that swap is a sequential comparison between two
models, not a concurrent one, so it cannot separate model identity from
anything else changing across calendar time.

    output/sec415_transition_contrast.csv
    output/sec415_weekly_around_transition.csv
"""
import pandas as pd

from common import (CRIT, W_BASE, W_PRE_TRANSITION, bh, clustered_contrast, col,
                    event_frame, ordinary, win, write)


def main():
    E = event_frame()
    O = ordinary(E)

    # ------------------------------------------- the transition contrast
    # Apr 8-14 is entirely gpt-4o-mini, Apr 15-23 entirely gpt-4.1-mini.
    pre, post = win(O, W_PRE_TRANSITION), win(O, W_BASE)
    rows = []
    for name in CRIT:
        c = col(name)
        r = clustered_contrast(pre, post, c)
        rows.append({"criterion": name,
                     "pre_mean": round(pre[c].mean(), 2),
                     "post_mean": round(post[c].mean(), 2),
                     "d": round(r["d"], 4), "lo": round(r["lo"], 4),
                     "hi": round(r["hi"], 4), "p": r["p"]})
    T = pd.DataFrame(rows)
    T["p_bh"] = bh(T.p)
    T.insert(1, "n_pre", len(pre))
    T.insert(2, "n_post", len(post))
    write(T, "sec415_transition_contrast.csv")

    core = T[T.criterion != "Holistic (1-5)"]
    print(f"  Apr 8-14 (gpt-4o-mini, n={len(pre)}) vs Apr 15-23 "
          f"(gpt-4.1-mini, n={len(post)}), ordinary traffic")
    print(f"  {int((core.p_bh < 0.05).sum())} of 6 core criteria survive BH")
    print(core[["criterion", "d", "lo", "hi", "p_bh"]].round(3).to_string(index=False))

    # --------------------------------- the weekly series across the swap
    # The pooled series does not move at the swap because the templated share
    # changes at the same time; the ordinary series does.
    v = col("Validation")
    weeks = []
    for label, d in (("all", E), ("ordinary", O)):
        g = d.set_index("ts")[v].resample("W-MON").agg(["mean", "size"])
        weeks.append(g["mean"].rename(f"validation_{label}"))
        weeks.append(g["size"].rename(f"turns_{label}"))
    share = (E.set_index("ts").templated.resample("W-MON").mean() * 100
             ).rename("templated_share_pct")
    W = pd.concat(weeks + [share], axis=1)
    W = W[(W.index >= "2025-03-24") & (W.index <= "2025-05-19")].round(3)
    W.index.name = "week_end"
    write(W.reset_index(), "sec415_weekly_around_transition.csv")
    print("\n  weekly validation around the transition:")
    print(W.to_string())


if __name__ == "__main__":
    main()
