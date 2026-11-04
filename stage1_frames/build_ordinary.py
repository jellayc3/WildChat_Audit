"""Build de-identified frames for the two balanced ordinary draws.

Same tail as build_frames.py -- opaque ids, the Hicke & Tomlinson template rule
applied with THIS package's definition rather than whatever the upstream CSV
carried, and the raw text dropped. The one difference is the score column: these
draws are annotated on validation only, with gemma-3-27b-it.

    python stage1_frames/build_ordinary.py

    data/ordinary_A_frame.parquet
    data/ordinary_B_frame.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA, HT_MIN_CONV, HT_PREFIX_CHARS, ORDINARY_FRAME,
                    RAW_ORDINARY, RELAY_ASST_MAX, RELAY_USER_MIN, SUF_27B)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_frames import add_time, opaque, sha256

CHUNK = 20_000
COLS = ["row_id", "ResponseId", "assistant_turn_index", "wildchat_model",
        "wildchat_language", "wildchat_timestamp", "wildchat_turn_count",
        "human_message_length_words", "assistant_message_length_words",
        "hashed_ip", "conversation_hash", "template_conv_n", "templated",
        "human_prev_message", "assistant_message",
        "validation" + SUF_27B]
RENAME = {"wildchat_model": "model", "wildchat_language": "language",
          "wildchat_turn_count": "turn_count",
          "validation" + SUF_27B: "validation"}


def build(draw):
    src = RAW_ORDINARY[draw]
    parts, first_key = [], {}
    seen = 0
    for ch in pd.read_csv(src, usecols=COLS, chunksize=CHUNK, low_memory=False):
        # the upstream flags are preserved under their own names before this
        # package recomputes the rule, so the two can be compared
        ch = ch.rename(columns={"templated": "templated_upstream",
                                "template_conv_n": "template_conv_n_upstream"})
        h = ch.human_prev_message.fillna("").astype(str)
        a = ch.assistant_message.fillna("").astype(str)
        ch["uw"] = h.str.split().str.len().astype("int32")
        ch["aw"] = a.str.split().str.len().astype("int32")
        # first 500 characters of the conversation's first user message,
        # first occurrence wins -- identical to build_frames.build_event
        for cid, k in zip(ch.conversation_hash, h.str.slice(0, HT_PREFIX_CHARS)):
            first_key.setdefault(cid, k)
        parts.append(ch.drop(columns=["human_prev_message", "assistant_message"]))
        seen += len(ch)
        print(f"\r  streamed {seen:,} rows", end="", flush=True)
    print()
    df = add_time(pd.concat(parts, ignore_index=True))
    del parts

    first_key = pd.Series(first_key)
    conv_n = first_key.value_counts()
    df["_k500"] = df.conversation_hash.map(first_key)
    df["template_conv_n"] = df._k500.map(conv_n).fillna(0).astype("int32")
    df["templated"] = df.template_conv_n >= HT_MIN_CONV
    shared = set(conv_n[conv_n > 1].index)
    df["template_id"] = opaque(df._k500.where(df._k500.isin(shared)),
                               sentinel=[np.nan])
    df["relay"] = (df.uw >= RELAY_USER_MIN) & (df.aw <= RELAY_ASST_MAX)

    df["conv_id"] = opaque(df.conversation_hash)
    df["sender_id"] = opaque(df.hashed_ip)
    df["cluster_id"] = opaque(df.conversation_hash.astype(str) + "|"
                              + df.hashed_ip.astype(str))
    df["judge"] = "gemma-3-27b-it"
    df["draw"] = draw

    df = df.drop(columns=[c for c in ("_k500", "conversation_hash", "hashed_ip",
                                      "row_id", "ResponseId")
                          if c in df.columns])
    df = df.rename(columns=RENAME)
    df.insert(0, "turn_id", np.arange(len(df), dtype="int32"))
    return df


def save(df, out):
    """Parquet where an engine is available, gzipped CSV otherwise.

    The distributable format is parquet; the fallback exists so the build runs
    inside environments without pyarrow (for example a Linux mount of this
    folder, where a macOS .venv is not executable).
    """
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception as exc:                                  # noqa: BLE001
        alt = out.replace(".parquet", ".csv.gz")
        df.to_csv(alt, index=False, compression="gzip")
        print(f"  ! parquet unavailable ({type(exc).__name__}); wrote {alt}")
        return alt


if __name__ == "__main__":
    lines = []
    for draw in sorted(RAW_ORDINARY):
        print(f"draw {draw}: {RAW_ORDINARY[draw]}")
        df = build(draw)
        scored = df.validation.notna()
        out = save(df, ORDINARY_FRAME[draw])
        agree = (df.templated == df.templated_upstream).mean()
        print(f"  {len(df):,} turns ({scored.sum():,} scored), "
              f"{df.cluster_id.nunique():,} clusters, "
              f"{df.conv_id.nunique():,} conversations -> {out}")
        print(f"  templated under THIS package's rule: "
              f"{df.templated.mean()*100:.2f}% of turns, "
              f"{df.groupby('conv_id').templated.first().mean()*100:.2f}% of "
              f"conversations, {df.loc[df.templated,'template_id'].nunique()} templates")
        print(f"  upstream flag agrees on {agree*100:.2f}% of turns "
              f"(upstream templated share {df.templated_upstream.mean()*100:.2f}%)")
        lines.append(f"{os.path.basename(out)}  {sha256(out)}")
    with open(os.path.join(DATA, "ORDINARY_CHECKSUMS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote data/ORDINARY_CHECKSUMS.txt")
