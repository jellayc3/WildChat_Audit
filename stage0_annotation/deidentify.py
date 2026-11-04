"""Strip message text and sender identifiers from a scored turn-level CSV.

Release step for the raw score files. Note that this is NOT what makes the
shipped package de-identified: stage 1 does that, and does it more thoroughly,
by replacing every identifier with an opaque integer. This script exists so the
raw files themselves can be released alongside, in the reduced form the earlier
`event_sample_turn_level.csv` took.

    python3 deidentify.py --in scored_with_ip.csv --out scored.csv
"""
import argparse

import pandas as pd

DROP = ["human_prev_message", "assistant_message", "hashed_ip",
        "conversation_hash", "conv_timestamp", "match_route"]

LEN_COLS = ["human_message_length_chars", "human_message_length_words",
            "assistant_message_length_chars", "assistant_message_length_words"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.src, low_memory=False)

    # Length columns must be computed before the text is dropped, otherwise
    # every downstream length analysis becomes impossible.
    if not all(c in df.columns for c in LEN_COLS) and "human_prev_message" in df.columns:
        h = df.human_prev_message.fillna("").astype(str)
        a = df.assistant_message.fillna("").astype(str)
        df["human_message_length_chars"] = h.str.len()
        df["human_message_length_words"] = h.str.split().str.len()
        df["assistant_message_length_chars"] = a.str.len()
        df["assistant_message_length_words"] = a.str.split().str.len()

    dropped = [c for c in DROP if c in df.columns]
    df = df.drop(columns=dropped)
    df.to_csv(args.out, index=False)

    print(f"dropped: {', '.join(dropped)}")
    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {args.out}")
    print("\nNote: without hashed_ip the clustering unit used throughout the "
          "paper cannot be reconstructed, and without message text the "
          "templated-traffic definition cannot be applied. Use the shipped "
          "stage-1 frames for analysis, not this file.")


if __name__ == "__main__":
    main()
