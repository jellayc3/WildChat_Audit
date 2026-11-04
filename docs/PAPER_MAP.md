# Paper → script → output

Section numbers follow the submitted paper. Every number listed here is checked by `stage4_verify/verify_paper.py`.

| Paper | Script | Output |
|---|---|---|
| Table 1 (samples, IPs) | `stage5_facct/f01_frames.py`, `stage2_analysis/t01_samples.py` | `output/facct/f01_samples.csv`, `output/table1_samples.csv` |
| §3.3 pool screen (2,808,223 conversations; 631 openings; 44.4%) | `stage1_frames/build_poolscreen.py` (needs the non-distributed pool counts) | pool totals are constants in `tools/build_appendix.py`; sample shares in `output/facct/f21_screen_shares.csv` |
| §3.4 judge comparison | `stage5_facct/f06_judges.py` | `output/facct/f06_judge_*.csv` |
| §3.5 weekly bins (turns per bin, final bin) | `stage5_facct/f21_replication_summaries.py` | `output/facct/f21_weekly_bins.csv` |
| §3.5 hash collisions | `stage2_analysis/t01_samples.py` | `output/sec35_hash_collisions.csv` |
| §4.1 all-traffic change, templated shares | `stage5_facct/f14_multicriteria.py` | `output/facct/f14_multicriteria.csv` |
| §4.1 sender concentration (39%) | `stage5_facct/f20_sender_concentration.py` | `output/facct/f20_sender_concentration.csv` |
| §4.1 per-template validation, dominant client | `stage2_analysis/sec415_transition.py` | `output/sec415_per_template_validation.csv`, `sec415_dominant_clients.csv` |
| §4.1 event window, changepoints, threshold sweep | `stage2_analysis/t05_changepoints.py`, `t06_event_contrasts.py`, `stage5_facct/f05_threshold.py` | `output/table5_*`, `table6_*`, `table7_*`, `table8_*`, `facct/f05_*` |
| Figure 2 | `stage5_facct/f09_figures.py` | `figures/facct/fig2_composition.png` |
| §4.2 main contrast, split-sample agreement | `stage5_facct/f02_ordinary.py` | `output/facct/f02_contrasts.csv`, `f02_replication.csv` |
| §4.2 post-rollback slope (turns) | `stage5_facct/f02_ordinary.py` | `output/facct/f02_postrollback_slope.csv` |
| §4.2 slope on weekly means; returning users (§4.3) | `stage5_facct/f19_within_user_weekly.py` | `output/facct/f19_weekly_hac.csv`, `f19_within_user*.csv` |
| §4.2 English / non-English, mix held fixed | `stage5_facct/f18_sensitivity.py` | `output/facct/f18_sensitivity.csv` |
| §4.2 changepoints | `stage5_facct/f04_changepoint_grid.py` | `output/facct/f04_changepoint_grid.csv` |
| §4.2 April dip and placebo | `stage5_facct/f16_dip.py`, `f17_placebo.py` | `output/facct/f16_dip_*.csv`, `f17_placebo.csv` |
| Figure 3 | `stage5_facct/f09_figures.py` | `figures/facct/fig3_ordinary_draws.png` |
| Table 3 | `f02_ordinary.py`, `f03_ordinary_grid.py`, `f18_sensitivity.py`, `f19_within_user_weekly.py` | `f02_*`, `f03_ordinary_grid*.csv`, `f18_sensitivity.csv`, `f19_*` |
| §4.4, Table 2 (12B criteria) | `stage5_facct/f14_multicriteria.py`, `f14b_rescreen_arms.py` | `output/facct/f14_multicriteria.csv`, `f14b_rescreen_arms.csv` |
| §5, §6.2 composition drift, serving concurrency | `f02_ordinary.py`, `f01_frames.py` | `f02_postrollback_composition.csv`, `f01_serving_overlaps.csv` |
| §7 near-threshold traffic | `f01_frames.py` | `output/facct/f01_near_threshold.csv` |
| Appendix | `tools/build_appendix.py` | `appendix/appendix.tex`, `appendix/figures/` |
| Every printed number | `stage4_verify/verify_paper.py` | `output/paper_claims_check.csv` |
