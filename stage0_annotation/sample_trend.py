"""Draw and score the TREND sample: 16,878 turns from 9,837 conversations,
Apr 2023 - Jul 2025.

Uniform random draw across the whole corpus with no date filter, which is what
gives it the full timeline and the concurrent model pairs of Section 4.2. Take
the first 10,000 shuffled conversations that contain at least one non-empty
assistant turn; the 163 shortfall to 9,837 is conversations failing that filter
after the fact.

    OPENROUTER_API_KEY=... python3 sample_trend.py --out ../raw/trend.csv

Roughly 16,878 turns x 8 criteria = 135,000 calls.
"""
import argparse
import os
import pickle

import pandas as pd

import prompts
import wildchat
from judge import score_all, SUF

N_CONVERSATIONS = 10_000
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="work_trend")
    ap.add_argument("--n", type=int, default=N_CONVERSATIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    turns_csv = os.path.join(args.work, "paired_turns.csv")
    if os.path.exists(turns_csv):
        turns = pd.read_csv(turns_csv)
        print(f"reusing {len(turns):,} turns from {turns_csv}")
    else:
        cache = os.path.join(args.work, "sample.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                sampled = pickle.load(fh)
        else:
            ds = wildchat.load_wildchat(args.seed)
            sampled = []
            for conv in ds:
                if wildchat.has_assistant_turn(conv):
                    sampled.append(conv)
                    if len(sampled) >= args.n:
                        break
            with open(cache, "wb") as fh:
                pickle.dump(sampled, fh)
        print(f"sampled {len(sampled):,} conversations")
        turns = wildchat.to_paired_turns(sampled)
        turns.to_csv(turns_csv, index=False)
        print(f"segmented into {len(turns):,} assistant turns")

    scores = score_all(turns, prompts.PROVENANCE["trend"], prompts.build_prompt,
                       os.path.join(args.work, "progress"))
    out = pd.concat([turns, scores], axis=1)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows -> {args.out}")
    print("score columns: " + ", ".join(c for c in out.columns if c.endswith(SUF)))


if __name__ == "__main__":
    main()
