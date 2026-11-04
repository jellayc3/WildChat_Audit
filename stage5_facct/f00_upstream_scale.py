"""Needs the raw CSVs (like stage 1). How large is the pool behind the ordinary draws'
upstream template counts? Compares, for openers seen in both, the upstream count shipped
with Draw A against the within-sample count in the event sample and in Draw A itself.
Also checks that the first row of each conversation in file order is its first turn.
    output/facct/f00_upstream_scale.csv
"""
import os, sys
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_EVENT, RAW_ORDINARY, HT_PREFIX_CHARS, OUTPUT

def firstkeys(path, extra=()):
    first, firstidx, minidx = {}, {}, {}
    for ch in pd.read_csv(path, usecols=["conversation_hash", "assistant_turn_index", "human_prev_message", *extra], chunksize=20000, low_memory=False):
        h = ch.human_prev_message.fillna("").astype(str).str.slice(0, HT_PREFIX_CHARS)
        for cid, k, ti, *ex in zip(ch.conversation_hash, h, ch.assistant_turn_index, *[ch[c] for c in extra]):
            if cid not in first:
                first[cid] = (k, *ex); firstidx[cid] = ti
            if ti < minidx.get(cid, 10 ** 9): minidx[cid] = ti
    df = pd.DataFrame.from_dict(first, orient="index", columns=["key", *extra])
    df["first_is_min"] = pd.Series(firstidx) == pd.Series(minidx)
    return df

def main():
    e = firstkeys(RAW_EVENT); e["ecount"] = e.key.map(e.key.value_counts())
    a = firstkeys(RAW_ORDINARY["A"], ["template_conv_n"]); a["acount"] = a.key.map(a.key.value_counts())
    ek = e.drop_duplicates("key").set_index("key").ecount
    j = a.drop_duplicates("key").set_index("key")[["template_conv_n", "acount"]].join(ek, how="inner")
    j = j[j.index.str.strip().str.len() > 0]
    rows = [dict(check="event: first row per conversation is its first turn", value=float(e.first_is_min.mean())),
            dict(check="Draw A: first row per conversation is its first turn", value=float(a.first_is_min.mean())),
            dict(check="Draw A: upstream count >= within-draw count (share of conversations)", value=float((a.template_conv_n >= a.acount).mean()))]
    for lo, hi in [(5, 19), (20, 49), (50, 99)]:
        s = j[(j.template_conv_n >= lo) & (j.template_conv_n <= hi)]
        rows += [dict(check=f"openers with upstream count {lo}-{hi}: n shared with event sample", value=len(s)),
                 dict(check=f"openers with upstream count {lo}-{hi}: median count within event sample", value=float(s.ecount.median())),
                 dict(check=f"openers with upstream count {lo}-{hi}: median count within Draw A", value=float(s.acount.median()))]
    out = os.path.join(OUTPUT, "facct"); os.makedirs(out, exist_ok=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "f00_upstream_scale.csv"), index=False); print(R.to_string())

if __name__ == "__main__":
    main()
