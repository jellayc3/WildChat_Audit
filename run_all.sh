#!/usr/bin/env bash
# Reproduce the paper's numbers, tables and figures.
#
#   ./run_all.sh              # check: rebuild the appendix and verify every printed number
#                             #        against the aggregate outputs shipped in output/
#   ./run_all.sh --analysis   # rerun stages 2 and 5 from the shipped score frames in data/
#   ./run_all.sh --frames     # build data/ from re-scored WildChat CSVs, then --analysis
#
# Stage 0 (sampling and LLM annotation) is never run here; see stage0_annotation/README.md.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$PWD/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplcache}"
"$PY" - <<'PYEOF' || exit 1
import sys
if sys.version_info < (3, 12):
    sys.exit(f"Python 3.12+ required (found {sys.version.split()[0]}); see README.md.")
PYEOF
run () { echo; echo "=== $* ==="; "$PY" "$@"; }

MODE="${1:-check}"
if [ "$MODE" = "check" ]; then
  run tools/build_appendix.py
  run stage4_verify/verify_paper.py
  exit $?
fi
for f in ordinary_A_frame.parquet ordinary_B_frame.parquet event_frame.parquet trend_frame.parquet; do
  if [ "$MODE" = "--analysis" ] && [ ! -f "data/$f" ]; then
    echo "data/$f not found. See README (Data)."; exit 1
  fi
done
if [ "$MODE" = "--frames" ]; then
  run stage1_frames/build_frames.py
  run stage1_frames/build_ordinary.py
  run stage1_frames/build_poolscreen.py
  run stage5_facct/f00_upstream_scale.py
fi
run stage1_frames/check_deidentification.py

cd stage2_analysis
for s in t01_samples t03_templated_profile t04_timeline t05_changepoints t06_event_contrasts \
         sec35_consensus_null sec415_transition multiverse_event t09_model_pairs pairs_multiverse; do
  run $s.py
done
cd ../stage5_facct
for s in f01_frames f02_ordinary f03_ordinary_grid f04_changepoint_grid f05_threshold f06_judges \
         f07_pairs f08_week_context f14_multicriteria f14b_rescreen_arms f16_dip f17_placebo \
         f18_sensitivity f19_within_user_weekly f20_sender_concentration f21_replication_summaries f09_figures; do
  run $s.py
done
cd ..
set +e
run tools/build_appendix.py
run stage4_verify/verify_paper.py
RC=$?
set -e
echo; echo "Done. CSVs in output/, figures in figures/facct/, claim check in output/paper_claims_check.csv"
exit $RC
