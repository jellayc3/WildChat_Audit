# Stage 0 — sampling and annotation

Draws the two 12B samples (supporting and trend) from WildChat and scores every turn
on the seven criteria. The 27B ordinary-traffic draws (A and B) were produced by a
separate pipeline that is not included; see the main README.
**This stage was not executed when the package was built**: the delivered score
files already exist, and re-running costs roughly 600,000 API calls. It ships
complete and runnable so the measurement instrument is inspectable and the
sampling is reproducible.

Stages 1–4 do not depend on this stage. They read the score CSVs it produces.

## What it needs

```bash
export HF_TOKEN=...              # allenai/WildChat-4.8M-Full is a gated dataset
export OPENROUTER_API_KEY=...
pip install -r ../requirements.txt
```

No GPU and no local model serving. Judging is entirely remote HTTP against
OpenRouter: `google/gemma-3-12b-it`, temperature 0, `max_tokens` 200, one call
per turn per criterion, 5 parallel workers, batches of 100 with a pickle
checkpoint written after each batch. An interrupted run resumes without
re-billing the work already done.

## Order

```bash
# 1. Trend sample: 10,000 conversations, uniform, no date filter -> 16,878 turns
python3 sample_trend.py --out ../raw/trend_sample_turn_level_with_ip.csv

# 2. Event sample, both runs. --exclude keeps it disjoint from the trend sample.
python3 sample_event.py --mode weekly_2025 \
    --out ../raw/event_2025.csv \
    --exclude ../raw/trend_sample_turn_level_with_ip.csv
python3 sample_event.py --mode weekly_nov_dec_2024 \
    --out ../raw/event_novdec.csv \
    --exclude ../raw/trend_sample_turn_level_with_ip.csv

# 3. Merge into the delivered column set -> 65,292 turns
python3 merge_samples.py --inputs ../raw/event_2025.csv ../raw/event_novdec.csv \
    --out ../raw/event_sample_turn_level_with_IP.csv

# 4. Optional: strip text and IP for public release
python3 deidentify.py --in ../raw/event_sample_turn_level_with_IP.csv \
    --out ../raw/event_sample_turn_level.csv
```

Scale: the trend sample is about 135,000 calls (16,878 turns × 8 criteria), the
event sample about 457,000 (65,292 × 7).

## Two things the original pipeline did not ship, reconstructed here

**The Nov–Dec 2024 sampling configuration.** The delivered event sample carries
two values in `source_file` — `ext_weekly2025` (48,752 turns) and
`ext_weekly_nov_dec_2024` (16,540) — but the source notebook defined only the
2025 configuration. The second is reconstructed in `sample_event.py` from the
delivered `source_file` and the earliest `sample_bucket`, `2024-W44`.

**The merge step.** No original script produced the delivered column set:
`source_file`, `sample_bucket_original` and the four message-length columns were
added somewhere between scoring and delivery. `merge_samples.py` is that step,
written from the delivered schema.

## The prompts are not from one file

Five criteria exist in two non-identical wordings and the paper's seven columns
are split across both. `prompts.PROVENANCE` is the authoritative record:

| criterion | trend sample | event sample |
|---|---|---|
| validation, agree, disagree, flatter, enthusiasm | "AI chatbot" wording | "AI assistant" wording |
| positive affirmation | "AI assistant" wording | "AI assistant" wording |
| holistic sycophancy (1–5) | Anthropic rubric | Anthropic rubric |

The "AI chatbot" wording comes from `wildchat_gemma_annotation_pipeline.py` and
is the only place the Anthropic 1–5 rubric lives; the "AI assistant" wording
comes from `wildchat_gemma_ext_annotation_colab.ipynb` and is the only place the
positive-affirmation prompt lives. A replication that re-scores under a single
wording should expect small level shifts in the five shared criteria, so
cross-sample comparisons of those five carry a wording difference as well as a
sampling one.

## Scales

Six criteria are 0–10, `0 = not at all` and `10 = very much`. The holistic
criterion is Anthropic's 1–5 rubric where **1 is extremely sycophantic and 5 is
no signs** — the opposite direction. Nothing in the package rescales it; the
sign is handled in prose and in figure captions.

## Keys

`row_id` is WildChat's own integer `turn_identifier` and is the only safe join
key. `ResponseId` is the content-derived `conversation_hash` and is **not
unique**: separate people who open with an identical message receive the same
hash. 93 hashes in the event sample carry five or more distinct senders and the
largest holds 583 turns from 583 senders. This is why conversations are
identified by `conversation_hash × hashed_ip` and not the hash alone; standard
errors in the paper are clustered on the hashed IP.
