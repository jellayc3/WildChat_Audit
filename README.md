# Auditing sycophancy indicators in naturalistic ChatGPT conversations

Replication package for the FAccT 2027 submission *What Can a Third-Party Auditor See? Limits of Auditing Sycophancy in Public Chat Logs* (anonymous review version).

It contains the analysis code, the LLM-judge scores behind every analysis, and the aggregate outputs behind every number, table and figure in the paper. `stage4_verify/verify_paper.py` checks each printed number against the outputs.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12+
./run_all.sh              # under a minute: rebuild the appendix and check every printed number
./run_all.sh --analysis   # 25-40 minutes: rerun every analysis from the scores in data/
```

Both end with the claim check; every row of `output/paper_claims_check.csv` should read `PASS`.

## Data

Judge outputs are stochastic, so rescoring WildChat with the same prompts will not reproduce our scores exactly. We therefore ship the scores themselves, so every analysis can be rerun on the same numbers we used.

| File | Rows | What it is |
|---|---|---|
| `data/ordinary_A_frame.parquet`, `data/ordinary_B_frame.parquet` | 80,468 / 79,474 | Two non-overlapping draws of ordinary (non-templated) conversations, Nov 2024 – Jul 2025, scored for validation by Gemma 3 27B |
| `data/event_frame.parquet` | 65,292 | Supporting sample, all traffic, same span, scored on six criteria and a holistic rating by Gemma 3 12B |
| `data/trend_frame.parquet` | 16,878 | Apr 2023 – Jul 2025 sample (12B), used for serving windows |
| `data/judge_overlap_turns.csv` | 6,630 | Turns scored by both judges |
| `data/greeting_conv.csv` | 306 | Supporting-sample conversations whose templated opener is a bare greeting (sensitivity arm) |
| `data/deployment_timeline.csv` | — | Public OpenAI release dates |

Each frame has one row per assistant turn: judge scores, timestamp, served model label, language, turn index, word counts, template flags, and opaque integer IDs.

**Identifiers.**
- `sender_id` is an opaque integer standing in for WildChat's hashed IP. It is kept because standard errors are clustered on it and the returning-user analysis depends on it. WildChat's hashed IP itself is not included.
- `conv_id`, `cluster_id` (conversation × sender) and `turn_id` are integers assigned within each frame, used only to group turns of the same conversation (cluster means, first turns). **WildChat's conversation ID (`conversation_hash`) and turn identifiers are not included**; the integers themselves carry no WildChat information.
- `template_id` is an opaque integer for a shared-opener group; the opener text is not included.
- IDs are assigned separately in each frame and do not link across frames.

No message text is included. `stage1_frames/check_deidentification.py` enforces all of this and runs before every analysis.

**Linkage.** Timestamps are kept at full resolution because the slopes are fit in continuous time; rounding them to the day changes several printed estimates. With access to WildChat, a row's timestamp, turn index and word counts are enough to find its conversation. That reveals nothing beyond what WildChat itself already publishes, but it means the frames should be treated as derived WildChat data under its terms, not as anonymous data.

**Provenance of Draws A and B.** The two ordinary-traffic draws were sampled and scored with Gemma 3 27B by a separate pipeline whose code is not included here; `stage0_annotation/` contains the sampling and scoring code for the 12B samples only (supporting and trend samples). The draws excluded templated conversations before sampling. That pipeline's per-turn template count is kept as `template_conv_n_upstream` / `templated_upstream`; its exact counting rule was not recorded. For 0.26% of Draw A conversations the within-draw count exceeds the recorded count (`output/facct/f00_upstream_scale.csv`). In the ordinary frames, `template_conv_n` / `templated` are recomputed within each draw with this package's definition; in `event_frame.parquet`, `template_conv_n` is the count over the WildChat pool from November 1, 2024.

**Not distributed:** the raw scored CSVs (message text, WildChat conversation IDs, hashed IPs) and the pool-level opener counts.

## Rebuilding from WildChat

The upstream corpus is [WildChat-4.8M-Full](https://huggingface.co/datasets/allenai/WildChat-4.8M-Full) (ODC-BY; flagged as containing sensitive content). To rescore from scratch:

1. **Sample and score (stage 0).** `stage0_annotation/` draws the 12B samples and scores turns with the prompts in `prompts.py` (about 600k judge calls for the 12B samples). Sampling is seeded; judge outputs are not, so rescored numbers will differ slightly from the paper. Draws A and B cannot be redrawn with this package (see Provenance above).
2. **Build frames (stage 1).** `./run_all.sh --frames`, with `SYCOPHANCY_RAW_DIR` pointing at the scored CSVs and `SYCOPHANCY_POOL_COUNTS` at the opener counts over the pool. This overwrites `data/` and then runs every analysis.

## Pipeline

| Stage | Folder | Role |
|---|---|---|
| 0 | `stage0_annotation/` | Sampling and LLM-judge scoring. Included for transparency and not run by `run_all.sh`. Re-scoring requires API access, at about 600k calls for the 12B samples. |
| 1 | `stage1_frames/` | Build turn-level frames from scored CSVs (local only); pool-level template screen; de-identification check |
| 2 | `stage2_analysis/` | Supporting-sample analyses: event-window contrasts, all-traffic changepoints, template profile, specification grid, serving windows |
| 5 | `stage5_facct/` | Ordinary-draw contrasts, slopes, 324-specification grid, changepoints, threshold sweep, judge comparison, six-criteria table, sensitivity checks, figures |
| 4 | `stage4_verify/` | `verify_paper.py`: every number printed in the paper against `output/` |
| — | `tools/` | `build_appendix.py`: writes the paper's appendix (LaTeX) from `output/` |

All windows, thresholds and estimator choices are declared once in `config.py` and `stage5_facct/fcommon.py`. `docs/PAPER_MAP.md` maps each section, table and figure to its script and output file.

## Estimators, in one paragraph

The unit of analysis is the assistant turn. Standard errors are clustered on the hashed sender IP, because turns from one user are not independent across conversations, and a few high-volume IPs dominate parts of the corpus (one produces 39% of post-rollback turns in the supporting sample). Contrasts and slopes are OLS with CR1 standard errors clustered on IP; the conversation × IP unit is used only as a unit of analysis (cluster means, first turns). Effect sizes are Cohen's d. Multiple testing is Benjamini–Hochberg within declared families (`docs/METHODS.md`). Changepoints are PELT (l2 cost, BIC-style penalty) on weekly means, with bootstrap support. Weekly bins run Tuesday–Monday and are labelled by their end date.

## License

Code: MIT (see `LICENSE`). Scores and aggregate outputs are derived from WildChat and subject to its license (ODC-BY) and terms of use; please cite WildChat alongside this work.
