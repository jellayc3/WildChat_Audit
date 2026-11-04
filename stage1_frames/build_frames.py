"""Build the de-identified analysis frames. The only stage that touches raw data.

Reads the two score CSVs, which carry `hashed_ip` and raw message text, derives
everything downstream needs from them, and then drops them. What ships is one
row per assistant turn with opaque integer identifiers and no free text:

    turn_id          sequential, not WildChat's turn_identifier
    cluster_id       conversation_hash x hashed_ip -- the unit of inference
    conv_id          conversation_hash alone, for the S2 dependence comparison
    sender_id        hashed_ip alone, for the distinct-sender counts in Table 3
    template_id      opaque id of the 500-char opener group, -1 if ungrouped
    template_conv_n  how many conversations share that opener
    templated        template_conv_n >= 100  (Hicke & Tomlinson)
    relay            user >= 500 words and assistant <= 60 words (weaker proxy)
    ts date week month
    model language turn_count assistant_turn_index
    uw aw            user / assistant word counts
    <seven score columns>

The 503 MB export is streamed in chunks and touched exactly once.

    python3 build_frames.py [--force]
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CRIT, DATA, EVENT_FRAME, HT_MIN_CONV, HT_PREFIX_CHARS,
                    RAW_EVENT, RAW_TREND, RELAY_ASST_MAX, RELAY_USER_MIN, SUF,
                    TREND_FRAME)

CHUNK = 20_000

KEEP_META = ["assistant_turn_index", "wildchat_model", "wildchat_language",
             "wildchat_turn_count"]
RENAME = {"wildchat_model": "model", "wildchat_language": "language",
          "wildchat_turn_count": "turn_count"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def opaque(series, sentinel=None):
    """Map arbitrary keys to sequential integers. Order is by first appearance
    so the ids are stable across reruns of the same input, and the mapping is
    one-way: nothing in the shipped frame can be turned back into a hash or an
    IP. `sentinel` values become -1."""
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    if sentinel is not None:
        codes = np.where(series.isin(sentinel).values, -1, codes)
    return codes.astype("int32")


def add_time(df):
    df["ts"] = pd.to_datetime(df.wildchat_timestamp, utc=True, errors="coerce").dt.tz_localize(None)
    df["date"] = df.ts.dt.normalize()
    df["week"] = df.ts.dt.to_period("W-SUN").dt.start_time
    df["month"] = df.ts.dt.to_period("M").dt.to_timestamp()
    return df.drop(columns=["wildchat_timestamp"])


def finish(df, first_key):
    """Shared tail: template grouping, the two proxies, opaque ids, cleanup.

    `first_key` maps conversation_hash -> the first 500 characters of that
    conversation's first user message. Hicke & Tomlinson count CONVERSATIONS
    sharing an opener, not turns, which is why the count is taken over that
    series rather than over the turn-level frame.
    """
    conv_n = first_key.value_counts()
    df["_k500"] = df.conversation_hash.map(first_key)
    df["template_conv_n"] = df._k500.map(conv_n).fillna(0).astype("int32")
    df["templated"] = df.template_conv_n >= HT_MIN_CONV

    # Only openers shared by more than one conversation form a group at all;
    # a unique opener gets -1 rather than a singleton id.
    shared = set(conv_n[conv_n > 1].index)
    df["template_id"] = opaque(df._k500.where(df._k500.isin(shared)), sentinel=[np.nan])

    df["relay"] = (df.uw >= RELAY_USER_MIN) & (df.aw <= RELAY_ASST_MAX)

    df["conv_id"] = opaque(df.conversation_hash)
    if "hashed_ip" in df.columns:
        df["sender_id"] = opaque(df.hashed_ip)
        df["cluster_id"] = opaque(df.conversation_hash.astype(str) + "|"
                                  + df.hashed_ip.astype(str))
    else:
        # The trend sample without IP falls back to the conversation. Recorded
        # so no downstream script silently assumes a sender is available.
        df["sender_id"] = -1
        df["cluster_id"] = df["conv_id"]

    df = df.drop(columns=[c for c in ("_k500", "conversation_hash", "hashed_ip",
                                      "human_prev_message", "assistant_message",
                                      "row_id", "ResponseId")
                          if c in df.columns])
    df.insert(0, "turn_id", np.arange(len(df), dtype="int32"))
    return df.rename(columns=RENAME)


def build_event():
    cols = (["row_id", "ResponseId", "wildchat_timestamp", "hashed_ip",
             "conversation_hash", "human_prev_message", "assistant_message"]
            + KEEP_META + [c + SUF for c in CRIT.values()])
    parts, first_key = [], {}
    seen = 0
    for ch in pd.read_csv(RAW_EVENT, usecols=cols, chunksize=CHUNK, low_memory=False):
        h = ch.human_prev_message.fillna("").astype(str)
        a = ch.assistant_message.fillna("").astype(str)
        ch["uw"] = h.str.split().str.len().astype("int32")
        ch["aw"] = a.str.split().str.len().astype("int32")
        for cid, k in zip(ch.conversation_hash, h.str.slice(0, HT_PREFIX_CHARS)):
            first_key.setdefault(cid, k)
        parts.append(ch.drop(columns=["human_prev_message", "assistant_message"]))
        seen += len(ch)
        print(f"\r  streamed {seen:,} rows", end="", flush=True)
    print()
    df = add_time(pd.concat(parts, ignore_index=True))
    del parts
    return finish(df, pd.Series(first_key))


def build_trend():
    df = pd.read_csv(RAW_TREND, low_memory=False)
    h = df.human_prev_message.fillna("").astype(str)
    df["uw"] = h.str.split().str.len().astype("int32")
    df["aw"] = df.assistant_message.fillna("").astype(str).str.split().str.len().astype("int32")
    df["conversation_hash"] = df.ResponseId
    first_key = (pd.DataFrame({"c": df.conversation_hash,
                               "k": h.str.slice(0, HT_PREFIX_CHARS)})
                 .groupby("c").k.first())
    df = add_time(df)
    return finish(df, first_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for path in (RAW_EVENT, RAW_TREND):
        if not os.path.exists(path):
            raise SystemExit(
                f"Raw input not found: {path}\n"
                "These files carry hashed_ip and message text and are not part "
                "of the distributable package. Point SYCOPHANCY_RAW_DIR at them, "
                "or skip stage 1 and use the shipped frames in data/.")

    checks = []
    for name, path, out, fn in (("event", RAW_EVENT, EVENT_FRAME, build_event),
                                ("trend", RAW_TREND, TREND_FRAME, build_trend)):
        if os.path.exists(out) and not args.force:
            print(f"{name}: {out} exists, skipping (--force to rebuild)")
        else:
            print(f"{name}: building from {os.path.basename(path)}")
            df = fn()
            df.to_parquet(out, index=False)
            print(f"  {len(df):,} turns, {df.cluster_id.nunique():,} clusters, "
                  f"{df.conv_id.nunique():,} conversations -> {out}")
        checks.append((name, os.path.basename(path), sha256(path)))

    with open(os.path.join(DATA, "INPUT_CHECKSUMS.txt"), "w") as fh:
        for name, base, digest in checks:
            fh.write(f"{name}\t{base}\t{digest}\n")
    print(f"checksums -> {os.path.join(DATA, 'INPUT_CHECKSUMS.txt')}")


if __name__ == "__main__":
    main()
