"""Every path, window, threshold, criterion name and estimator choice, declared once.

No script in this package defines any of these locally. Section, table and
figure numbers in comments refer to the analysis history of the project; the
map from the submitted paper to scripts is docs/PAPER_MAP.md.
"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(PKG, "data")       # de-identified, shipped
OUTPUT  = os.path.join(PKG, "output")
FIGURES = os.path.join(PKG, "figures")
DOCS    = os.path.join(PKG, "docs")
for _d in (DATA, OUTPUT, FIGURES, DOCS):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ raw inputs
# Stage 1 only. These carry hashed_ip and raw message text and are NOT part of
# the distributable package; stage1_frames reads them once and emits the
# de-identified frames in data/. Override with SYCOPHANCY_RAW_DIR.
RAW_DIR = os.environ.get(
    "SYCOPHANCY_RAW_DIR",
    os.path.join(PKG, "raw"))
RAW_EVENT = os.path.join(RAW_DIR, "event_sample_turn_level_with_IP.csv")   # 503 MB
RAW_TREND = os.path.join(RAW_DIR, "trend_sample_turn_level_with_ip.csv")   # 111 MB

EVENT_FRAME = os.path.join(DATA, "event_frame.parquet")
TREND_FRAME = os.path.join(DATA, "trend_frame.parquet")

# ------------------------------------------------------------------ criteria
# Column stem -> the suffix the annotation stage writes.
SUF = "_google_gemma_3_12b_it_score"

# Display name -> column stem. This order is the order used in every table.
CRIT = {
    "Validation":           "validation",
    "Agree":                "agree",
    "Disagree":             "disagree",
    "Flatter":              "flatter",
    "Enthusiasm":           "enthusiasm",
    "Positive affirmation": "bot-positive-affirmation",
    "Holistic (1-5)":       "sycophancy",
}
CORE6 = [k for k in CRIT if k != "Holistic (1-5)"]

# The holistic criterion is scored 1-5 with 1 = extreme sycophancy, i.e. the
# opposite direction from the six 0-10 criteria. Nothing rescales it; the sign
# is handled in prose and in figure captions.
HOLISTIC = "Holistic (1-5)"

# ------------------------------------------------------------------ windows
# Half-open [start, end). "Apr 15-23 inclusive" is ("2025-04-15", "2025-04-24");
# the captions were off by a day once already.
W_BASE  = ("2025-04-15", "2025-04-24")   # Apr 15-23, n = 2,077
W_EVENT = ("2025-04-25", "2025-04-28")   # Apr 25-27, n =   442
W_POST  = ("2025-05-01", "2025-05-15")   # May  1-14, n = 4,054

# Section 4.1.5 contrasts the two models either side of the Apr 15 transition:
# Apr 8-14 is entirely gpt-4o-mini, Apr 15-23 entirely gpt-4.1-mini.
W_PRE_TRANSITION = ("2025-04-08", "2025-04-15")

# ------------------------------------------------------------ traffic definitions
# THE definition (paper 3.5), Hicke & Tomlinson [2026] verbatim: a conversation
# is templated when at least HT_MIN_CONV conversations in the dataset open with
# the same first HT_PREFIX_CHARS characters of the user's first message.
# Counts are taken over the WildChat-4.8M-Full pool from Nov 1, 2024 onward
# (2,808,223 conversations; 631 openers reach 100, covering 44.4%). The shipped
# event frame carries these pool counts in template_conv_n; the earlier
# within-sample counts are kept as template_conv_n_within for comparison.
HT_PREFIX_CHARS = 500
HT_MIN_CONV     = 100

# The length signature. Section 4.1.5 calls this the weaker proxy and the
# supplement's earlier draft used it as its primary definition; it is kept so
# the supplement rewrite can state old value -> new value, and so the recall
# check against it can be reported. It is not a treatment in any grid.
RELAY_USER_MIN, RELAY_ASST_MAX = 500, 60

# RETIRED, and no script may reintroduce them as treatments or grid arms:
#   prevalence-matched (>=2 conversations)  flags 43.3% of turns and 88.5% of
#     the event window; the only cutoff at which the validation contrast changes
#     sign, where >=10 through >=100 all give +0.08 to +0.15.
#   300-char opener seen >=20 times         superseded in-house rule.
#   H&T sender-level filter                 removed 21.7% of the sample but 0%
#     of the event window, so it separated nothing.

# ------------------------------------------------------------------ estimators
# Standard errors are clustered on the hashed sender IP (opaque int). Turns from one IP
# are not independent across conversations, and a few high-volume IPs (one sender produces
# 39% of post-rollback turns in the supporting sample) make conversation-level clustering
# too optimistic. The conversation x IP unit (cluster_id) is kept as a unit of analysis
# (cluster means, first turn per cluster), never as the clustering unit for inference.
CLUSTER_UNIT = "sender_id"
UNIT_ID      = "cluster_id"   # conversation_hash x hashed_ip, for unit-of-analysis arms
CONV_UNIT    = "conv_id"      # conversation_hash alone, for the S2 comparison
BOOTSTRAP_DRAWS = 2000
PERM_DRAWS      = 10000
BH_FAMILY_EVENT = 14          # 7 criteria x {event, post}
BH_FAMILY_PAIRS = 56          # 7 criteria x 8 pairs

# The event-contrast specification grid (3.5, 4.1.4, 4.1.5):
#   4 baselines x 2 event windows x 2 traffic treatments x 3 units
#   x 3 length adjustments = 144. Full factorial, no cell dropped.
#
# The three units are the turn, the CLUSTER mean and the first turn per CLUSTER,
# where the cluster is CLUSTER_UNIT above. It is not the conversation:
# conversation_hash collides across senders (3.5), which is why the cluster
# exists at all.
#
# The length covariate is log1p(words). Under the cluster-mean unit it is
# aggregated as the within-cluster MEAN OF THE LOGS, never the log of the mean,
# which would silently fit a different covariate from the one the other two
# units use.
N_SPECS = 144
GRID_BASELINES = {
    "Apr15-23 (paper)": ("2025-04-15", "2025-04-24"),
    "Apr16-24":         ("2025-04-16", "2025-04-25"),
    "Apr15-21":         ("2025-04-15", "2025-04-22"),
    "Apr19-24":         ("2025-04-19", "2025-04-25"),
}
GRID_EVENTS = {
    "Apr25-27 (paper)": ("2025-04-25", "2025-04-28"),
    "Apr22-27":         ("2025-04-22", "2025-04-28"),   # includes the pre-announcement climb
}
# A cell is skipped only when a window holds fewer than this many usable turns,
# which never happens on the arms above; the guard exists so a future arm cannot
# silently fit a degenerate regression.
GRID_MIN_N = 30
GRID_UNITS   = ["turn", "cluster_mean", "cluster_first"]
GRID_LENGTHS = [None, "user", "user+assistant"]
GRID_TRAFFIC = ["pooled", "templated_removed"]

# The concurrent model-pair grid (3.5, 4.2.2):
#   3 overlap thresholds x 3 volume thresholds x 3 clustering units
#   x the same 2 traffic treatments = 54. The two grids never combine.
N_SPECS_PAIRS = 54
GRID_PAIR_OVERLAP = [14, 21, 28]        # days of serving-window overlap
GRID_PAIR_VOLUME  = [60, 120, 200]      # minimum turns per model inside it
# The clustering unit is itself a researcher choice, so it is an arm too:
# conversation alone, conversation x sender, and sender alone.
GRID_PAIR_UNITS   = ["conv_id", "cluster_id", "sender_id"]

PAIR_MIN_OVERLAP_DAYS = 21    # "at least three overlapping weeks"
PAIR_MIN_TURNS        = 120   # "at least 120 turns per model during their overlap"
PAIR_EXCLUDE = {"o1-preview-2024-09-12", "o1-mini-2024-09-12"}  # different product class

# ------------------------------------------------------------------ changepoints
# ONE changepoint analysis. Weekly bins are resampled W-MON and LABELLED BY
# THEIR END DATE, every non-empty week is a bin, and the penalty is BIC-style:
#     pen = log(n) * var(diff(y)) / 2
# The bin labelled 2025-04-28 therefore covers April 22-27: the climb plus the
# whole event window. A tuned 0.25*var(diff) penalty with Monday-anchored bins
# and a >=30-turn minimum was used in an earlier rebuild and is retired; its bin
# labelled 2025-04-28 covered May 1-4 and held no event-window data at all.
PELT_MIN_SIZE  = 2
PELT_RESAMPLE  = "W-MON"
WEEK_MIN_TURNS = 0

# Minimum bin occupancy for the figures. These affect what is DRAWN and never
# what is estimated, so they live here rather than inside the plotting code.
DAILY_MIN_TURNS = 15    # Figure 5, stated in its caption
MONTH_MIN_TURNS = 3     # Figure 7, stated in its caption
MODEL_MIN_TURNS = 30    # Figure 8
CONSENSUS_K    = 2            # ">= 2 of 6 core criteria"
NULL_DRAWS     = 2000         # correlation-preserving week-order permutations
BOOT_SUPPORT_DRAWS = 500

SEED = 11
PERM_SEED = 42                # the permutation nulls carry their own fixed seed


# =========================================================== ordinary draws (2026-09)
# Two balanced weekly draws of ORDINARY traffic only: templated conversations are
# excluded at the sampling stage rather than filtered afterwards, ~1,000
# conversations per week over the event sample's span. Draw B shares no
# conversation and no turn with Draw A (non-overlapping draws from one pool).
#
# These are scored with gemma-3-27b-it, the variant Rathje et al. validated, on
# the validation criterion only. Levels are therefore NOT comparable with the
# 12B frames above; shapes and shifts are. Every table must name its judge.
SUF_27B = "_google_gemma_3_27b_it_score"
ORDINARY_DIR = os.environ.get(
    "SYCOPHANCY_ORDINARY_DIR",
    RAW_DIR)
RAW_ORDINARY = {
    "A": os.path.join(ORDINARY_DIR,
         "turn_level_scores_wildchat_ordinary_ordinary_weekly_event_combined.csv"),
    "B": os.path.join(ORDINARY_DIR,
         "turn_level_scores_wildchat_ordinary_ordinary_weekly_event_combined_2_validation_complete.csv"),
}
ORDINARY_FRAME = {k: os.path.join(DATA, f"ordinary_{k}_frame.parquet")
                  for k in RAW_ORDINARY}

# The templated flag shipped in those CSVs was computed by the pipeline that drew
# the samples; its exact counting rule was not recorded (see README, Provenance
# of Draws A and B). build_ordinary.py recomputes it with THE definition above so the frames
# are comparable with the event frame, and records both.

# ------------------------------------------------------- April decomposition windows
# Four windows for the Kitagawa decomposition (kitagawa_windows.py). W_KIT_2 is
# W_BASE and W_KIT_3 is W_EVENT widened to Apr 28, which adds no turns because
# the corpus carries nothing for Apr 28-30 -- kept so the window reads as the
# calendar period rather than as a data-availability artifact.
W_KIT_1 = ("2025-04-06", "2025-04-15")   # pre-switch,  entirely gpt-4o-mini
W_KIT_2 = ("2025-04-15", "2025-04-24")   # post-switch, = W_BASE
W_KIT_3 = ("2025-04-25", "2025-04-29")   # disclosed event window
W_KIT_4 = ("2025-05-01", "2025-05-09")   # post-rollback
KIT_WINDOWS = {"W1 Apr 6-14 (pre-switch)": W_KIT_1,
               "W2 Apr 15-23 (post-switch)": W_KIT_2,
               "W3 Apr 25-28 (event)": W_KIT_3,
               "W4 May 1-8 (post-event)": W_KIT_4}
# A decomposition term is reported as identified only when the two windows'
# group distributions overlap by at least this much, sum_g min(s1_g, s2_g).
KIT_MIN_SUPPORT = 0.10
