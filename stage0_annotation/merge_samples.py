"""Merge the two event-sample runs into the delivered column set.

This step existed only implicitly: no script in the original pipeline produced
the exact columns found in the delivered CSV. The two runs were concatenated,
`source_file` and `sample_bucket_original` were added, and the four message
length columns were computed. That is what this reconstructs.

    python3 merge_samples.py --inputs ../raw/event_2025.csv ../raw/event_novdec.csv \
                             --out ../raw/event_sample_turn_level_with_IP.csv

Expected result: 65,292 rows, 48,752 from ext_weekly2025 and 16,540 from
ext_weekly_nov_dec_2024.
"""
import argparse

import pandas as pd

from judge import SUF
from prompts import PAPER_CRITERIA

# Column order of the delivered file.
BASE_COLS = ["row_id", "ResponseId", "assistant_turn_index", "wildchat_model",
             "wildchat_language", "wildchat_timestamp", "wildchat_toxic",
             "wildchat_turn_count", "sample_bucket"]
SCORE_COLS = [c + SUF for c in PAPER_CRITERIA]
LEN_COLS = ["human_message_length_chars", "human_message_length_words",
            "assistant_message_length_chars", "assistant_message_length_words"]
TEXT_COLS = ["human_prev_message", "assistant_message"]
ID_COLS = ["hashed_ip", "conversation_hash"]
TAIL_COLS = ["source_file", "sample_bucket_original"]


def add_lengths(df):
    h = df.human_prev_message.fillna("").astype(str)
    a = df.assistant_message.fillna("").astype(str)
    df["human_message_length_chars"] = h.str.len()
    df["human_message_length_words"] = h.str.split().str.len()
    df["assistant_message_length_chars"] = a.str.len()
    df["assistant_message_length_words"] = a.str.split().str.len()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = []
    for p in args.inputs:
        d = pd.read_csv(p, low_memory=False)
        print(f"{p}: {len(d):,} rows")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # The notebook rewrote sample_bucket during scoring; the delivered file
    # keeps the original alongside it.
    if "sample_bucket_original" not in df.columns:
        df["sample_bucket_original"] = df["sample_bucket"]

    df = add_lengths(df)

    # row_id is WildChat's turn_identifier and is the only safe join key:
    # ResponseId is content-derived and collides across senders.
    dupes = df.row_id.duplicated().sum()
    if dupes:
        print(f"warning: {dupes:,} duplicated row_id; dropping the later copy")
        df = df.drop_duplicates("row_id", keep="first")

    cols = [c for c in BASE_COLS + SCORE_COLS + LEN_COLS + TEXT_COLS + ID_COLS + TAIL_COLS
            if c in df.columns]
    df = df[cols]
    df.to_csv(args.out, index=False)

    print(f"\nwrote {len(df):,} rows -> {args.out}")
    print(df.source_file.value_counts().to_string())


if __name__ == "__main__":
    main()
