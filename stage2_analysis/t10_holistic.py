"""Table 10: the holistic Anthropic rating across every analysis reported above.

A second instrument on the same data. Scored 1-5 with 1 the most sycophantic,
so its sign is inverted relative to the six 0-10 criteria and nothing rescales
it. Reads the artifacts the other scripts already produced rather than
recomputing them, so Table 10 cannot drift from the tables it summarises.

    output/table10_holistic.csv
"""
import os

import pandas as pd

from common import N_SPECS, N_SPECS_PAIRS, out_path, write

H = "Holistic (1-5)"


def need(name):
    p = out_path(name)
    if not os.path.exists(p):
        raise SystemExit(f"{name} missing -- run the script that produces it first")
    return pd.read_csv(p)


def main():
    t6 = need("table6_event_contrasts.csv").set_index("criterion")
    t7 = need("table7_contrasts_templated_removed.csv").set_index("criterion")
    mv = need("multiverse_by_criterion.csv").set_index("criterion")
    pairs = need("table9_model_pairs.csv").set_index("criterion")
    pm = need("pairs_multiverse.csv") if os.path.exists(
        out_path("pairs_multiverse.csv")) else None
    corr = need("sec43_length_correlations.csv").set_index("criterion")

    e, p = t6.loc[H], t7.loc[H]
    rows = [
        {"what_we_measured": "Event window (Apr 25-27 vs Apr 15-23), all traffic",
         "result": f"{e.pre:.2f} -> {e.event:.2f}, d = {e.event_d:+.2f} "
                   f"[{e.event_lo:+.2f}, {e.event_hi:+.2f}]"},
        {"what_we_measured": "Event window, templated conversations removed",
         "result": f"d = {p.event_d:+.2f} [{p.event_lo:+.2f}, {p.event_hi:+.2f}] "
                   f"- sign reversal, null"},
        {"what_we_measured": "Post-rollback (May 1-14)",
         "result": f"{e.post:.2f}, d = {e.post_d:+.2f} [{e.post_lo:+.2f}, {e.post_hi:+.2f}]"},
        {"what_we_measured": f"Sign changes across the {N_SPECS}-specification grid",
         "result": f"{mv.loc[H, 'pct_sign_flip']:.1f}% "
                   f"(disagreement: {mv.loc['Disagree', 'pct_sign_flip']:.1f}%)"},
        {"what_we_measured": "Separation between concurrently served models",
         "result": f"{int(pairs.loc[H, 'all_traffic_bh'])} of "
                   f"{int(pairs.loc[H, 'all_traffic_n_pairs'])} pairs on all traffic"
                   + (f", median {pm[pm.criterion == H].median_pairs.iloc[0]:.0f} "
                      f"(range {pm[pm.criterion == H].min_pairs.iloc[0]:.0f}-"
                      f"{pm[pm.criterion == H].max_pairs.iloc[0]:.0f}) across the "
                      f"54-specification grid" if pm is not None else "")},
        {"what_we_measured": "Correlation with assistant reply length",
         "result": f"r = {corr.loc[H, 'r_log_assistant_words']:+.2f} against log reply "
                   f"length, the strongest of any criterion"},
    ]
    T = pd.DataFrame(rows)
    write(T, "table10_holistic.csv")
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
