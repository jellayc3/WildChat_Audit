"""Assert that nothing shippable carries an identifier or free text.

Run after build_frames.py and again before releasing the package. Exits non-zero
on the first violation.

    python3 check_deidentification.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA, EVENT_FRAME, TREND_FRAME
ORDINARY = [os.path.join(DATA, f"ordinary_{k}_frame.parquet") for k in "AB"]

# Columns that must never appear in a shipped frame.
BANNED = {"hashed_ip", "conversation_hash", "turn_identifier", "ResponseId", "row_id",
          "human_prev_message", "assistant_message", "conv_timestamp",
          "_k500", "k500", "tmpl", "match_route"}

# Any object-dtype column must be drawn from a small controlled vocabulary
# (model strings, language names). Anything with many distinct long values is
# free text that escaped the drop.
MAX_DISTINCT = 500
MAX_LEN = 80


def check_frame(path):
    problems = []
    df = pd.read_parquet(path)
    name = os.path.basename(path)

    for col in df.columns:
        if col in BANNED:
            problems.append(f"{name}: banned column {col!r}")

    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        if vals.nunique() > MAX_DISTINCT:
            problems.append(
                f"{name}: {col!r} has {vals.nunique():,} distinct string values, "
                f"which looks like free text rather than a category")
        longest = vals.str.len().max()
        if longest > MAX_LEN:
            problems.append(
                f"{name}: {col!r} contains a {longest}-character string, "
                f"longer than any model or language label")

    # The identifiers that do ship must be opaque integers.
    for col in ("cluster_id", "conv_id", "sender_id", "template_id", "turn_id"):
        if col not in df.columns:
            problems.append(f"{name}: missing {col!r}")
        elif not pd.api.types.is_integer_dtype(df[col]):
            problems.append(f"{name}: {col!r} is {df[col].dtype}, expected an integer")

    return df, problems


def main():
    all_problems = []
    for path in (EVENT_FRAME, TREND_FRAME, *ORDINARY):
        if not os.path.exists(path):
            all_problems.append(f"missing frame: {path}")
            continue
        df, problems = check_frame(path)
        all_problems += problems
        print(f"{os.path.basename(path)}: {len(df):,} rows x {len(df.columns)} cols, "
              f"{df.cluster_id.nunique():,} clusters, "
              f"{df.sender_id.nunique():,} senders, "
              f"{(df.template_id >= 0).sum():,} turns in a shared-opener group")

    J = os.path.join(DATA, "judge_overlap_turns.csv")
    if os.path.exists(J):
        j = pd.read_csv(J)
        bad = set(j.columns) & BANNED
        if bad:
            all_problems.append(f"judge_overlap_turns.csv: banned columns {sorted(bad)}")

    # Nothing in data/ may be one of the raw inputs, however it got there.
    for f in os.listdir(DATA):
        if f.endswith(".csv") and os.path.getsize(os.path.join(DATA, f)) > 50_000_000:
            all_problems.append(f"data/{f} is large enough to be a raw export")

    if all_problems:
        print("\nFAIL")
        for p in all_problems:
            print("  " + p)
        sys.exit(1)
    print("\nPASS: no identifiers or free text in any shipped frame")


if __name__ == "__main__":
    main()
