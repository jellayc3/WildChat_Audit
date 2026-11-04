# Methods reference

This summarizes the analysis choices declared in `config.py` and `stage5_facct/fcommon.py`. It does not replace the paper's methods section.

## Units

- **Analysed unit:** the assistant turn.
- **Independence unit (standard errors):** the hashed sender IP, shipped as the opaque integer `sender_id`.
- **Conversation unit:** conversation hash × hashed IP (`cluster_id`), used for cluster-mean and first-turn specifications. WildChat's conversation hash is derived from content, so different users who open with the same text share a hash. In the supporting sample, 93 hashes carry five or more senders, and the largest holds 583 turns from 583 senders.

## Traffic

A conversation is **templated** when at least 100 conversations share the first 500 characters of its first user message (Hicke & Tomlinson). Counts are taken over the WildChat-4.8M-Full pool from Nov 1, 2024 (2,808,223 conversations). Draws A and B exclude every conversation at or above 100 before sampling. In the supporting sample, `template_conv_n` holds the pool count; stricter arms use thresholds of 20 and 5.

## Windows (half-open)

| Name | Dates | Used for |
|---|---|---|
| Before | Nov 1, 2024 – Apr 14, 2025 | era contrast |
| After | Apr 29 – Jul 31, 2025 (the corpus is empty Apr 28–30) | era contrast |
| Post-rollback slope | May 1 – Jul 31, 2025 | weekly slope; one model label throughout |
| Baseline / event | Apr 15–23 / Apr 25–27 | disclosed-update window |

## Estimators

- **Contrasts and slopes:** OLS with CR1 standard errors clustered on `sender_id` (hashed IP). Covariates, where used: log(1 + reply words), log(1 + user words), language indicators, turn index.
- **Effect size:** Cohen's d with turn-level pooled SD.
- **Weekly intervals in figures:** intercept-only OLS per week, clustered as above.
- **Weekly-means trend:** OLS on the 12 complete weekly means (bins ending May 5–July 28), slope per calendar week, Newey–West standard errors with 2 (and 4) lags.
- **Returning users:** IPs present before April 15 and after the rollback; fixed-effect regression with one intercept per IP (turn-weighted), and the mean of per-IP differences (each IP weighted equally).
- **Specification grid (324 per draw):** 4 before windows × 3 after windows × 3 units (turn, cluster mean, first turn per cluster) × 3 length adjustments × 3 template thresholds.
- **Multiple testing:** Benjamini–Hochberg within each draw's 324 specifications; within the seven criteria of each column of the six-criteria table; within the seven criteria of each event-window arm. The main era contrast and slope are single pre-specified tests.
- **Changepoints:** PELT with l2 cost, minimum segment of 2 weekly bins, penalty log(n)·var(diff(y))/2. Support is the share of 500 parametric resamples of the weekly means (mean + N(0, turn-level SE)) in which a break recurs within 14 days. This resampling does not account for clustering.
- **Weekly bins:** `W-MON`, running Tuesday–Monday and labelled by their end date. The last bin holds Jul 29–31.

## Judges

- Draws A and B: `gemma-3-27b-it`, validation only, 0–10.
- Supporting and trend samples: `gemma-3-12b-it`, six 0–10 criteria plus a 1–5 holistic rating (1 = most sycophantic).

The two judges are not on a common scale. Compare shapes and shifts, never levels.
