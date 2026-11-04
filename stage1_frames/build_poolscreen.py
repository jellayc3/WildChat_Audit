"""Re-count the Hicke & Tomlinson templated rule for the event sample over the full
WildChat-4.8M-Full pool (Nov 1, 2024 onward), the same pool that screened Draws A and B,
instead of within the 37,997-conversation event sample.

Needs two inputs that are NOT distributed:
  SYCOPHANCY_RAW_DIR/event_sample_turn_level_with_IP.csv   raw scored event sample
  SYCOPHANCY_POOL_COUNTS                                   pickle: {'counts': {first-500-char opener: n conversations in pool}}
Writes data/event_frame.parquet (pool counts in template_conv_n; the within-sample
values kept as *_within) and data/greeting_conv.csv. The shipped frame is the output of
this script, so reviewers do not need to run it.
"""
import os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from config import RAW_EVENT, EVENT_FRAME, DATA, HT_PREFIX_CHARS, HT_MIN_CONV
from build_frames import opaque

POOL = os.environ.get("SYCOPHANCY_POOL_COUNTS")
GREET = re.compile(r"^\s*(hi+|hello+|hey+|hola|ola|привет|здравствуйте|سلام|مرحبا|你好|selam|merhaba|bonjour|ciao|hallo|olá|how are you|hi,? how are you|test)\W*$", re.I)

def main():
    if not (POOL and os.path.exists(POOL) and os.path.exists(RAW_EVENT)):
        sys.exit("build_poolscreen: raw event CSV or SYCOPHANCY_POOL_COUNTS not found; the shipped frame is already pool-screened.")
    P = pd.read_pickle(POOL)["counts"]
    E = pd.read_parquet(EVENT_FRAME)
    if "template_conv_n_within" in E.columns:
        sys.exit("build_poolscreen: data/event_frame.parquet is already pool-screened.")
    cols = ["conversation_hash", "human_prev_message", "validation_google_gemma_3_12b_it_score"]
    fk, hashes, vals = {}, [], []
    for ch in pd.read_csv(RAW_EVENT, usecols=cols, chunksize=20000, low_memory=False):
        h = ch.human_prev_message.fillna("").astype(str)
        for cid, k in zip(ch.conversation_hash, h.str.slice(0, HT_PREFIX_CHARS)): fk.setdefault(cid, k)
        hashes.append(ch.conversation_hash.values); vals.append(ch["validation_google_gemma_3_12b_it_score"].values)
    hashes = np.concatenate(hashes); v = np.concatenate(vals)
    assert len(hashes) == len(E), "row count mismatch"
    assert np.allclose(np.nan_to_num(v, nan=-9), np.nan_to_num(E["validation_google_gemma_3_12b_it_score"].values, nan=-9)), "row order mismatch"
    assert (pd.factorize(hashes)[0] == E.conv_id.values).all(), "conversation order mismatch"
    k = pd.Series(hashes).map(pd.Series(fk))
    full = k.map(lambda s: P.get(s, 0)).astype("int32").values
    E["template_conv_n_within"] = E.template_conv_n; E["templated_within"] = E.templated; E["template_id_within"] = E.template_id
    E["template_conv_n"] = full; E["templated"] = full >= HT_MIN_CONV
    E["template_id"] = opaque(k.where(pd.Series(full > 1)), sentinel=[np.nan])
    E.to_parquet(EVENT_FRAME, index=False)
    g = k.map(lambda s: bool(GREET.match(s))) & pd.Series(E.templated.values)
    pd.DataFrame({"conv_id": np.unique(E.conv_id.values[g.values])}).to_csv(os.path.join(DATA, "greeting_conv.csv"), index=False)
    c = E.groupby("conv_id")[["templated_within", "templated"]].first()
    print("turns templated: within %.2f%% -> pool %.2f%%" % (100 * E.templated_within.mean(), 100 * E.templated.mean()))
    print("convs templated: within %.2f%% -> pool %.2f%%" % (100 * c.templated_within.mean(), 100 * c.templated.mean()))

if __name__ == "__main__":
    main()
