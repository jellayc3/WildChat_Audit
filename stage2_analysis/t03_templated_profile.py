"""Table 3: templated against ordinary conversations across the event sample,
on the Hicke & Tomlinson definition.

Also emits the template inventory behind Section 4.1.5 -- how many templates,
what share of conversations, per-template validation dispersion, and the
dominant client in each of the three windows.

    output/table3_templated_profile.csv
    output/sec415_template_inventory.csv
    output/sec415_dominant_clients.csv
"""
import pandas as pd

from common import (CORE6, HT_MIN_CONV, HT_PREFIX_CHARS, W_BASE, W_EVENT,
                    W_POST, col, event_frame, win, write)


def profile(d):
    out = {
        "turns": len(d),
        "share_of_turns_pct": None,          # filled by caller
        "clusters": d.cluster_id.nunique(),
        "conversations": d.conv_id.nunique(),
        "distinct_senders": d.sender_id.nunique(),
        "user_words_median": d.uw.median(),
        "user_words_mean": round(d.uw.mean(), 0),
        "user_words_sd": round(d.uw.std(), 0),
        "assistant_words_median": d.aw.median(),
        "assistant_words_mean": round(d.aw.mean(), 0),
        "assistant_words_sd": round(d.aw.std(), 0),
    }
    for c in CORE6:
        out[f"{c}_mean"] = round(d[col(c)].mean(), 2)
        out[f"{c}_sd"] = round(d[col(c)].std(), 2)
    return out


def main():
    E = event_frame()

    # ------------------------------------------------------------- Table 3
    rows = []
    for label, d in (("Ordinary", E[~E.templated]), ("Templated", E[E.templated])):
        r = profile(d)
        r["share_of_turns_pct"] = round(100 * len(d) / len(E), 1)
        rows.append({"traffic": label, **r})
    write(pd.DataFrame(rows), "table3_templated_profile.csv")

    # ------------------------------------------- template inventory (4.1.5)
    # A template is an opener group meeting the H&T threshold. The count the
    # paper quotes is over CONVERSATIONS, not turns.
    conv = E.drop_duplicates("conv_id")
    tmpl = conv[conv.templated]
    inv = pd.DataFrame([{
        "prefix_chars": HT_PREFIX_CHARS,
        "min_conversations": HT_MIN_CONV,
        "n_templates": tmpl.template_id.nunique(),
        "templated_conversations": len(tmpl),
        "total_conversations": len(conv),
        "templated_conversation_share_pct": round(100 * len(tmpl) / len(conv), 1),
        "templated_turn_share_pct": round(100 * E.templated.mean(), 1),
    }])
    write(inv, "sec415_template_inventory.csv")

    # Per-template validation, which is the dispersion the paper quotes as
    # running from 0.01 to 9.62.
    per = (E[E.templated].groupby("template_id")
           .agg(turns=("turn_id", "size"),
                conversations=("conv_id", "nunique"),
                validation_mean=(col("Validation"), "mean"))
           .reset_index()
           # template_id breaks ties so the file is byte-stable across runs.
           .sort_values(["turns", "template_id"], ascending=[False, True])
           .reset_index(drop=True))
    per["validation_mean"] = per.validation_mean.round(2)
    write(per, "sec415_per_template_validation.csv")

    # ------------------------------------------- dominant client per window
    rows = []
    for label, w in (("baseline Apr15-23", W_BASE), ("event Apr25-27", W_EVENT),
                     ("post May1-14", W_POST)):
        d = win(E, w)
        t = d[d.templated]
        if t.empty:
            continue
        top = t.template_id.value_counts()
        tid = top.index[0]
        rows.append({
            "window": label,
            "window_turns": len(d),
            "templated_turns": len(t),
            "templated_share_pct": round(100 * len(t) / len(d), 1),
            "n_templates_present": t.template_id.nunique(),
            "dominant_template_id": int(tid),
            "dominant_turns": int(top.iloc[0]),
            "dominant_share_of_templated_pct": round(100 * top.iloc[0] / len(t), 1),
            "dominant_share_of_window_pct": round(100 * top.iloc[0] / len(d), 1),
            "dominant_validation_mean": round(t[t.template_id == tid][col("Validation")].mean(), 2),
        })
    write(pd.DataFrame(rows), "sec415_dominant_clients.csv")


if __name__ == "__main__":
    main()
