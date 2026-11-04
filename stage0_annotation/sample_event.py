"""Draw and score the EVENT sample: 65,292 turns, Nov 2024 - Jul 2025, dense
weekly draw of ~1,000 conversations per ISO week.

The delivered event sample came from TWO runs, not one:

    ext_weekly2025             48,752 turns
    ext_weekly_nov_dec_2024    16,540 turns
                               ------
                               65,292

Only the 2025 configuration shipped with the source notebook; the Nov-Dec 2024
configuration below was reconstructed from the delivered `source_file` and
`sample_bucket` columns, whose earliest bucket is 2024-W44. Run both, then
merge with merge_samples.py.

    OPENROUTER_API_KEY=... python3 sample_event.py --mode weekly_2025      --out ../raw/event_2025.csv
    OPENROUTER_API_KEY=... python3 sample_event.py --mode weekly_nov_dec_2024 --out ../raw/event_novdec.csv

Roughly 65,292 turns x 7 criteria = 457,000 calls, so budget accordingly and
rely on the checkpointing rather than trying to finish in one sitting.
"""
import argparse
import os
import pickle
from collections import defaultdict

import pandas as pd

import prompts
import wildchat
from judge import score_all, SUF

SEED = 42

SAMPLING_CONFIGS = {
    # Shipped with the source notebook.
    "weekly_2025": {
        "n_per_bucket": 1000, "bucket": "week",
        "start": "2025-01-01", "end": "2025-12-31", "label": "ext_weekly2025",
    },
    # RECONSTRUCTED. Not in the source notebook; inferred from the delivered
    # source_file value and the 2024-W44 earliest bucket.
    "weekly_nov_dec_2024": {
        "n_per_bucket": 1000, "bucket": "week",
        "start": "2024-11-01", "end": "2024-12-31",
        "label": "ext_weekly_nov_dec_2024",
    },
    # Shipped but unused by the paper; kept for completeness.
    "daily_window": {
        "n_per_bucket": 1000, "bucket": "day",
        "start": "2025-04-21", "end": "2025-05-07", "label": "daily_apr21_may07",
    },
}


def bucket_of(ts, kind):
    """ISO week (Monday-Sunday, UTC) or calendar day."""
    if kind == "week":
        iso = ts.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return ts.strftime("%Y-%m-%d")


def sample_stratified(ds, cfg, seen_ids):
    """One pass over the shuffled dataset, filling each bucket up to
    n_per_bucket. Buckets that never fill are kept short with a warning rather
    than dropped, which is why some weeks carry fewer than 1,000 conversations.
    """
    start = pd.Timestamp(cfg["start"], tz="UTC")
    end = pd.Timestamp(cfg["end"], tz="UTC") + pd.Timedelta(days=1)
    n_per = cfg["n_per_bucket"]

    buckets = defaultdict(list)
    for conv in ds:
        if str(conv.get("conversation_hash")) in seen_ids:
            continue
        ts = wildchat.parse_ts(conv)
        if pd.isna(ts) or not (start <= ts < end):
            continue
        if not wildchat.has_assistant_turn(conv):
            continue
        b = bucket_of(ts, cfg["bucket"])
        if len(buckets[b]) >= n_per:
            continue
        conv = dict(conv)
        conv["_sample_bucket"] = b
        buckets[b].append(conv)

    short = [b for b, v in buckets.items() if len(v) < n_per]
    if short:
        print(f"  {len(short)} buckets short of {n_per}: {', '.join(sorted(short)[:8])}"
              + (" ..." if len(short) > 8 else ""))
    return [c for b in sorted(buckets) for c in buckets[b]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(SAMPLING_CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="CSVs whose ResponseIds must not be re-drawn "
                         "(pass the trend sample to guarantee disjoint samples)")
    args = ap.parse_args()

    cfg = SAMPLING_CONFIGS[args.mode]
    work = args.work or f"work_event_{args.mode}"
    os.makedirs(work, exist_ok=True)

    turns_csv = os.path.join(work, "paired_turns.csv")
    if os.path.exists(turns_csv):
        turns = pd.read_csv(turns_csv)
        print(f"reusing {len(turns):,} turns from {turns_csv}")
    else:
        cache = os.path.join(work, "sample.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                sampled = pickle.load(fh)
        else:
            seen = wildchat.seen_hashes(*args.exclude)
            print(f"excluding {len(seen):,} already-sampled conversation hashes")
            ds = wildchat.load_wildchat(args.seed)
            sampled = sample_stratified(ds, cfg, seen)
            with open(cache, "wb") as fh:
                pickle.dump(sampled, fh)
        print(f"sampled {len(sampled):,} conversations")
        turns = wildchat.to_paired_turns(sampled)
        bmap = {c["conversation_hash"]: c["_sample_bucket"] for c in sampled}
        turns["sample_bucket"] = turns.ResponseId.map(bmap)
        turns["source_file"] = cfg["label"]
        turns.to_csv(turns_csv, index=False)
        print(f"segmented into {len(turns):,} assistant turns")

    scores = score_all(turns, prompts.PROVENANCE["event"], prompts.build_prompt,
                       os.path.join(work, "progress"))
    out = pd.concat([turns, scores], axis=1)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows -> {args.out}")
    print("score columns: " + ", ".join(c for c in out.columns if c.endswith(SUF)))


if __name__ == "__main__":
    main()
