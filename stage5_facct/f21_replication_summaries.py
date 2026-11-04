"""Aggregate summaries that let stage4_verify and tools/build_appendix run
without the turn-level frames (which are not distributed).

Writes
  output/facct/f21_weekly_bins.csv   per draw: mean scored turns per non-empty
                                     W-MON bin (Nov 1 - Jul 31), last bin end, last turn date
  output/facct/f21_screen_shares.csv supporting sample: share of conversations and
                                     turns flagged as templated by the pool count
                                     and by a within-sample count
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA, OUTPUT

F = os.path.join(OUTPUT, "facct"); os.makedirs(F, exist_ok=True)

rows = []
for k in ("A", "B"):
    d = pd.read_parquet(os.path.join(DATA, f"ordinary_{k}_frame.parquet"))
    d = d[d.validation.notna()]
    w = d[(d.ts >= "2024-11-01") & (d.ts < "2025-08-01")].set_index("ts").validation.resample("W-MON").size()
    w = w[w > 0]
    rows.append(dict(draw=k, nonempty_bins=len(w), mean_turns_per_bin=w.mean(),
                     last_bin_end=w.index.max().date(), last_turn_date=d.ts.max().date()))
pd.DataFrame(rows).to_csv(os.path.join(F, "f21_weekly_bins.csv"), index=False)

E = pd.read_parquet(os.path.join(DATA, "event_frame.parquet"))
conv = E.groupby("conv_id")[["templated", "templated_within"]].first()
pd.DataFrame([
    dict(screen="pool", pct_conversations=100 * conv.templated.mean(), pct_turns=100 * E.templated.mean()),
    dict(screen="within_sample", pct_conversations=100 * conv.templated_within.mean(), pct_turns=100 * E.templated_within.mean()),
]).to_csv(os.path.join(F, "f21_screen_shares.csv"), index=False)
print(open(os.path.join(F, "f21_weekly_bins.csv")).read()); print(open(os.path.join(F, "f21_screen_shares.csv")).read())
