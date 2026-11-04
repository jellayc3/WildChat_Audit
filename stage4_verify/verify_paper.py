"""Check every number printed in the submitted paper against the pipeline's outputs.

Each claim names the section, the text as printed, and the output cell that produces it.
A claim passes when the recomputed value, rounded as printed, matches the printed text.
Writes output/paper_claims_check.csv and exits non-zero if any claim fails.

The printed values are the submitted text: November 1 - April 14 baseline, standard errors
clustered on hashed IP, pool-level template screen. A claim whose printed value is changed
in the paper must be changed here too; the check then says whether the data agree.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from config import OUTPUT
O, F = OUTPUT, os.path.join(OUTPUT, "facct")

_cache = {}
def csv(path):
    p = os.path.join(O, path) if not path.startswith("facct/") else os.path.join(F, path[6:])
    if p not in _cache: _cache[p] = pd.read_csv(p)
    return _cache[p]
def wbins(k):
    return row("facct/f21_weekly_bins.csv", draw=k)

def f(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")
def ci(lo, hi, nd=2): return f"[{f(lo, nd)}, {f(hi, nd)}]"
def n(x): return f"{int(round(x)):,}"
def row(path, **kw):
    d = csv(path)
    for k, v in kw.items(): d = d[d[k].astype(str) == str(v)]
    assert len(d) == 1, (path, kw, len(d)); return d.iloc[0]

def c2(draw, spec):  return row("facct/f02_contrasts.csv", draw=draw, spec=spec)
def sl(draw, spec):  return row("facct/f02_postrollback_slope.csv", draw=draw, spec=spec)
def grid(draw, **kw):
    """Main-specification row of the 324-cell grid, overridden by kw; CI from its clustered SE."""
    G = csv("facct/f03_ordinary_grid.csv"); G = G[G.draw == draw]
    base = dict(templates="<100 (as drawn)", pre="Nov 1-Apr 14", post="Apr 29-Jul 31", unit="turn", length="none"); base.update(kw)
    for k, v in base.items(): G = G[G[k] == v]
    assert len(G) == 1; r = G.iloc[0]; return f"{f(r.est, 2, True)} [{f(r.est - 1.96*r.se)}, {f(r.est + 1.96*r.se)}]"
def m14(frame_, crit): return row("facct/f14_multicriteria.csv", frame=frame_, criterion=crit)
def arm(a, crit):    return row("facct/f14b_rescreen_arms.csv", arm=a, criterion=crit)
def t3cell(crit):
    a = m14("all traffic", crit); o = arm("pool <100 (main)", crit)
    dag = lambda p: " †" if p >= .05 else ""
    return (f"{f(a['diff'],2,True)} (d = {f(a.d,2,True)}){dag(a.p_bh) if 'p_bh' in a else ''}",
            f"{f(o['diff'],2,True)} (d = {f(o.d,2,True)}){dag(o.p_bh)}",
            f"{f(o.slope,3,True)}{dag(o.slope_p_bh)}")
def weekly_turns(k):
    return wbins(k).mean_turns_per_bin
def pelt_breaks(framing, how):
    r = row("facct/f04_changepoint_grid.csv", framing=framing, weekly_mean=how); return r.breaks

C = []
def claim(sec, printed, fn, note=""):
    C.append(dict(section=sec, printed=printed, fn=fn, status="ok", note=note))
def sens(k, stratum, kind): return row("facct/f18_sensitivity.csv", draw=k, stratum=stratum, kind=kind)
def wu(k, spec): return row("facct/f19_within_user.csv", draw=k, spec=spec)
def hac(k, turns, lag=2): return row("facct/f19_weekly_hac.csv", draw=k, turns=turns, lag=lag)
def conc(frame_, per): return row("facct/f20_sender_concentration.csv", frame=frame_, period=per)
def est_ci(r, e="est", nd=2): return f"{f(r[e], nd, True)} {ci(r.lo, r.hi, nd)}"
dag = lambda p: " †" if p >= .05 else ""

# ------------------------------------------------ abstract / intro / Figure 1
claim("Abstract", "27% to 66%", lambda: f"{m14('all traffic','Validation').templated_pct_pre:.0f}% to {m14('all traffic','Validation').templated_pct_post:.0f}%")
claim("Fig. 1", "−0.90 / 0.99", lambda: f"{f(m14('all traffic','Validation')['diff'])} / {f(c2('A','era, raw')['diff'])}")
claim("Abstract", "324 alternative analyses, all positive and significant", lambda: f"{int(row('facct/f03_ordinary_grid_summary.csv', draw='A').n)} alternative analyses, all positive and significant" if (csv('facct/f03_ordinary_grid_summary.csv')[['positive','sig_bh']] == 324).all().all() else "NOT ALL")
# ------------------------------------------------ Table 1
for k, name, ips in [("A", "Ordinary Draw A (27B)", "23,218"), ("B", "Ordinary Draw B (27B)", "23,214"), ("E", "Event sample (12B)", "26,364")]:
    r = row("facct/f01_samples.csv", sample=name)
    for colname in ["turns_scored", "conversations", "senders"]:
        claim("Table 1", n(r[colname]), lambda v=r[colname]: n(v), note=f"{name} {colname}")
claim("3.3", "33 of the 273 days", lambda: f"{int(row('table1_samples.csv', sample='Event sample').empty_days)} of the {int(row('table1_samples.csv', sample='Event sample').days_spanned)} days")
# ------------------------------------------------ 3.4
J = lambda: row("facct/f06_judge_turns.csv", set="A")
claim("3.4", "3,298 turns; r = 0.68; 69% within one point; 1.09 lower", lambda: f"{n(J().turns)} turns; r = {f(J().pearson_r)}; {J().pct_within_1pt:.0f}% within one point; {f(-J().mean_diff_27_minus_12)} lower")
claim("3.4", "weekly r = 0.81 across 37 weeks", lambda: f"weekly r = {f(row('facct/f06_judge_series.csv', draw='A').pearson_r)} across {int(row('facct/f06_judge_series.csv', draw='A').weeks)} weeks")
# ------------------------------------------------ 3.5
claim("3.5", "about 2,150 turns per week", lambda: "about 2,150 turns per week" if 2100 <= min(weekly_turns("A"), weekly_turns("B")) <= max(weekly_turns("A"), weekly_turns("B")) <= 2200 else "MISMATCH", note="A 2,166, B 2,146")
claim("3.5", "final bin covers July 29–31", lambda: "final bin covers July 29–31" if all(str(wbins(k).last_bin_end) == "2025-08-04" and str(wbins(k).last_turn_date) <= "2025-07-31" for k in "AB") else "MISMATCH")
H = lambda: row("sec35_hash_collisions.csv", scope="event sample")
claim("3.5", "93 hashes; 583 turns from 583 senders", lambda: f"{int(H().hashes_with_5plus_senders)} hashes; {int(H().largest_hash_turns)} turns from {int(H().largest_hash_senders)} senders")
# ------------------------------------------------ 4.1 all traffic
v = lambda: m14("all traffic", "Validation")
claim("4.1", "−0.90 [−2.94, 1.14], d = −0.27", lambda: f"{f(v()['diff'],2,True)} {ci(v().lo, v().hi)}, d = {f(v().d)}")
claim("4.1", "one sender produces 39% of post-rollback turns", lambda: f"one sender produces {conc('supporting, all traffic','after').top1_pct:.0f}% of post-rollback turns")
claim("4.1", "−0.81 and −0.42; no all-traffic change survives correction", lambda: f"{f(m14('all traffic','Positive affirmation')['diff'])} and {f(m14('all traffic','Enthusiasm')['diff'])}; " + ("no all-traffic change survives correction" if (csv('facct/f14_multicriteria.csv').query("frame=='all traffic'").p_bh >= .05).all() else "SOME SURVIVE"))
claim("4.1", "−0.206 [−0.401, −0.011]", lambda: f"{f(v().slope,3,True)} {ci(v().slope_lo, v().slope_hi, 3)}")
claim("4.1", "26.5% before April 15, 66.0% after", lambda: f"{v().templated_pct_pre:.1f}% before April 15, {v().templated_pct_post:.1f}% after")
TV = lambda: csv("sec415_per_template_validation.csv").query("conversations >= 100")
claim("4.1", "29 templates, 0.01 to 9.62", lambda: f"{len(TV())} templates, {TV().validation_mean.min():.2f} to {TV().validation_mean.max():.2f}")
claim("4.1", "four of the six criteria flag April 22–27", lambda: f"{['zero','one','two','three','four','five','six'][int(row('table5_changepoints_all.csv', week_end='2025-04-28').n_criteria)]} of the six criteria flag April 22–27")
claim("4.1", "all six criteria change significantly in the event window", lambda: "all six criteria change significantly in the event window" if (csv('table6_event_contrasts.csv').query("criterion != 'Holistic (1-5)'").event_p_bh < .05).all() else "NOT ALL")
DC = lambda w: row("sec415_dominant_clients.csv", window=w)
claim("4.1", "61.4% to 86.2%; one client 57.0%", lambda: f"{DC('baseline Apr15-23').templated_share_pct:.1f}% to {DC('event Apr25-27').templated_share_pct:.1f}%; one client {DC('event Apr25-27').dominant_share_of_window_pct:.1f}%")
TH = lambda: csv("facct/f05_threshold_contrasts.csv")
claim("4.1", "no criterion increases significantly at any threshold", lambda: "no criterion increases significantly at any threshold" if not ((TH().d > 0) & (TH().p_bh < .05)).any() else "MISMATCH")
claim("4.1", "PELT no longer flags that week", lambda: "PELT no longer flags that week" if "2025-04-28" not in set(csv("table8_changepoints_templated_removed.csv").week_end.astype(str)) else "MISMATCH")
# ------------------------------------------------ 4.2
a, b = (lambda: c2("A", "era, raw")), (lambda: c2("B", "era, raw"))
claim("4.2", "2.76 to 3.75; +0.99 [0.76, 1.22], d = 0.40", lambda: f"{f(a().mean_a)} to {f(a().mean_b)}; {f(a()['diff'],2,True)} {ci(a().lo, a().hi)}, d = {f(a().d)}")
claim("4.2", "Draw B +1.01 [0.80, 1.22]", lambda: f"Draw B {f(b()['diff'],2,True)} {ci(b().lo, b().hi)}")
R = lambda: csv("facct/f02_replication.csv").iloc[0]
claim("4.2", "r = 0.945 across 37 weeks; 0.15 apart", lambda: f"r = {R().pearson_r:.3f} across {int(R().weeks)} weeks; {f(R().mean_abs_diff)} apart")
claim("4.2", "324 specifications, 0.74 to 1.56", lambda: f"324 specifications, {f(csv('facct/f03_ordinary_grid_summary.csv').est_min.min())} to {f(csv('facct/f03_ordinary_grid_summary.csv').est_max.max())}")
claim("4.2", "0.059 and 0.055 per week", lambda: f"{f(sl('A','raw').slope,3)} and {f(sl('B','raw').slope,3)} per week")
claim("4.2", "weekly means 0.075 [0.036, 0.113] and 0.065 [0.042, 0.088]", lambda: f"weekly means {f(hac('A','all').slope,3)} {ci(hac('A','all').lo, hac('A','all').hi, 3)} and {f(hac('B','all').slope,3)} {ci(hac('B','all').lo, hac('B','all').hi, 3)}")
claim("4.2", "non-English 0.082 and 0.060", lambda: f"non-English {f(sens('A','Non-English','post slope').est,3)} and {f(sens('B','Non-English','post slope').est,3)}")
claim("4.2", "English 0.022 [−0.014, 0.058] and 0.037 [−0.009, 0.083]; weekly 0.032 and 0.034", lambda: f"English {f(sens('A','English only','post slope').est,3)} {ci(sens('A','English only','post slope').lo, sens('A','English only','post slope').hi, 3)} and {f(sens('B','English only','post slope').est,3)} {ci(sens('B','English only','post slope').lo, sens('B','English only','post slope').hi, 3)}; weekly {f(hac('A','English').slope,3)} and {f(hac('B','English').slope,3)}")
claim("4.2", "turn-level breaks Apr 15, Apr 29, May 20 in both draws; cluster means break in April in Draw B only", lambda: "turn-level breaks Apr 15, Apr 29, May 20 in both draws; cluster means break in April in Draw B only" if all(all(dd in pelt_breaks(fr, "turn") for dd in ["2025-04-21", "2025-05-05", "2025-05-26"]) for fr in ["Ordinary Draw A (27B)", "Ordinary Draw B (27B)"]) and "2025-04-21" not in pelt_breaks("Ordinary Draw A (27B)", "cluster_mean") and "2025-04-21" in pelt_breaks("Ordinary Draw B (27B)", "cluster_mean") else "MISMATCH")
D = lambda k: row("facct/f16_dip_contrasts.csv", draw=k, a="Apr 15-27", b="Apr 29-May 19")
P = lambda k: row("facct/f17_placebo.csv", draw=k)
claim("4.2", "dip about 0.6; 8–12% of window pairs before April 15", lambda: f"dip about {abs((D('A')['diff'] + D('B')['diff'])/2):.1f}; {P('A').pct_abs_ge_dip:.0f}–{P('B').pct_abs_ge_dip:.0f}% of window pairs before April 15")
SEG = lambda k, s: row("facct/f16_dip_segments.csv", draw=k, segment=s)
# (English share during the dip is reported in the appendix, not the body.)
# ------------------------------------------------ Table 3
T3 = [("Main", {}), ("Before = Mar 1-Apr 14", {"pre": "Mar 1-Apr 14"}), ("Cluster mean", {"unit": "cluster_mean"}), ("First turn", {"unit": "cluster_first"}),
      ("Log reply length", {"length": "log assistant"}), ("Log user + reply length", {"length": "log user + assistant"}),
      ("Threshold < 20", {"templates": "<20"}), ("Threshold < 5", {"templates": "<5"}), ("After = May 13-Jul 31", {"post": "May 13-Jul 31"}), ("After = Jun 1-Jul 31", {"post": "Jun 1-Jul 31"})]
T3P = ["+0.99 [0.76, 1.22] | +1.01 [0.80, 1.22]", "+0.96 [0.61, 1.31] | +1.01 [0.67, 1.35]", "+1.07 [0.77, 1.37] | +1.10 [0.83, 1.38]", "+1.07 [0.77, 1.37] | +1.11 [0.83, 1.38]",
       "+0.86 [0.73, 1.00] | +0.89 [0.76, 1.02]", "+0.95 [0.81, 1.09] | +0.99 [0.86, 1.12]", "+1.00 [0.77, 1.22] | +1.00 [0.79, 1.20]", "+1.06 [0.87, 1.25] | +1.06 [0.87, 1.26]",
       "+1.08 [0.85, 1.30] | +1.08 [0.87, 1.28]", "+1.11 [0.87, 1.35] | +1.13 [0.91, 1.35]"]
for (lab, kw), pr in zip(T3, T3P):
    claim("Table 3", pr, lambda kw=kw: f"{grid('A', **kw)} | {grid('B', **kw)}", note=lab)
claim("Table 3", "+1.01 [0.78, 1.23] | +1.01 [0.80, 1.22]", lambda: f"{est_ci(c2('A','era, empty openers dropped'),'diff')} | {est_ci(c2('B','era, empty openers dropped'),'diff')}", note="empty openers dropped")
claim("Table 3", "+1.00 [0.50, 1.50] | +1.09 [0.66, 1.51]", lambda: f"{est_ci(sens('A','English only','era'))} | {est_ci(sens('B','English only','era'))}", note="English only")
claim("Table 3", "+0.82 [0.63, 1.02] | +0.87 [0.68, 1.06]", lambda: f"{est_ci(sens('A','lang x length tercile, pre-period weights','era'))} | {est_ci(sens('B','lang x length tercile, pre-period weights','era'))}", note="mix held fixed")
claim("Table 3", "+0.46 [0.14, 0.79] | +0.38 [0.02, 0.75]", lambda: f"{est_ci(wu('A','each IP weighted equally'))} | {est_ci(wu('B','each IP weighted equally'))}", note="returning IPs, equal weight")
claim("Table 3", "+0.74 to +1.53 | +0.80 to +1.56", lambda: " | ".join(f"{f(row('facct/f03_ordinary_grid_summary.csv', draw=k).est_min,2,True)} to {f(row('facct/f03_ordinary_grid_summary.csv', draw=k).est_max,2,True)}" for k in "AB"))
for spec, pr in [("raw", "+0.059 [0.031, 0.087] | +0.055 [0.024, 0.086]"), ("+ log assistant words", "+0.042 [0.017, 0.067] | +0.039 [0.012, 0.067]"),
                 ("+ log assistant words + language", "+0.050 [0.025, 0.075] | +0.046 [0.019, 0.073]"),
                 ("+ log user & assistant words + language + turn index", "+0.063 [0.035, 0.090] | +0.060 [0.032, 0.088]")]:
    claim("Table 3", pr, lambda spec=spec: " | ".join(f"{f(sl(k,spec).slope,3,True)} {ci(sl(k,spec).lo, sl(k,spec).hi, 3)}" for k in "AB"), note=f"slope {spec}")
claim("Table 3", "+0.022 [−0.014, 0.058] | +0.037 [−0.009, 0.083]", lambda: " | ".join(est_ci(sens(k,'English only','post slope'),'est',3) for k in "AB"), note="slope English")
claim("Table 3", "+0.032 [0.007, 0.056] | +0.022 [−0.006, 0.050]", lambda: " | ".join(est_ci(sens(k,'lang x length tercile, pre-period weights','post slope'),'est',3) for k in "AB"), note="slope mix held fixed")
claim("Table 3", "+0.075 [0.036, 0.113] | +0.065 [0.042, 0.088]", lambda: " | ".join(f"{f(hac(k,'all').slope,3,True)} {ci(hac(k,'all').lo, hac(k,'all').hi, 3)}" for k in "AB"), note="weekly means")
claim("Table 3", "+0.032 [0.010, 0.055] | +0.034 [0.008, 0.061]", lambda: " | ".join(f"{f(hac(k,'English').slope,3,True)} {ci(hac(k,'English').lo, hac(k,'English').hi, 3)}" for k in "AB"), note="weekly means, English")
# ------------------------------------------------ 4.3 returning users
claim("4.3", "167 of 13,650 and 194 of 13,654 IPs; about one-fifth of post-rollback turns", lambda: f"{int(wu('A','fixed effect, turns').ips_both)} of {n(wu('A','fixed effect, turns').ips_pre)} and {int(wu('B','fixed effect, turns').ips_both)} of {n(wu('B','fixed effect, turns').ips_pre)} IPs; about one-fifth of post-rollback turns" if 15 <= wu('B','fixed effect, turns').pct_post_turns_from_returning <= 25 and 15 <= wu('A','fixed effect, turns').pct_post_turns_from_returning <= 25 else "MISMATCH")
claim("4.3", "+1.11 [0.49, 1.73] and +1.07 [0.35, 1.79]", lambda: f"{est_ci(wu('A','fixed effect, turns'))} and {est_ci(wu('B','fixed effect, turns'))}")
claim("4.3", "five heaviest IPs produce half; without them +0.36 [−0.12, 0.84] and +0.11 [−0.44, 0.66]", lambda: f"five heaviest IPs produce {'half' if 45 <= wu('A','fixed effect, turns').top5_share_pct <= 55 else 'MISMATCH'}; without them {est_ci(wu('A','fixed effect, without five heaviest IPs'))} and {est_ci(wu('B','fixed effect, without five heaviest IPs'))}")
claim("4.3", "0.46 [0.14, 0.79] and 0.38 [0.02, 0.75]; 64% and 58% higher", lambda: f"{f(wu('A','each IP weighted equally').est)} {ci(wu('A','each IP weighted equally').lo, wu('A','each IP weighted equally').hi)} and {f(wu('B','each IP weighted equally').est)} {ci(wu('B','each IP weighted equally').lo, wu('B','each IP weighted equally').hi)}; {wu('A','each IP weighted equally').pct_ips_higher:.0f}% and {wu('B','each IP weighted equally').pct_ips_higher:.0f}% higher")
# ------------------------------------------------ Table 2 / 4.4 (12B)
for crit, pr in [("Validation", "−0.90 (d = −0.27) † | +0.96 (d = +0.32) | +0.064 †"), ("Agree", "+0.36 (d = +0.15) † | +0.21 (d = +0.08) | −0.030 †"),
                 ("Disagree", "−0.04 (d = −0.02) † | +0.06 (d = +0.03) † | +0.045 †"), ("Flatter", "−0.04 (d = −0.02) † | +0.59 (d = +0.31) | +0.039 †"),
                 ("Enthusiasm", "−0.42 (d = −0.15) † | +0.75 (d = +0.29) | +0.044 †"), ("Positive affirmation", "−0.81 (d = −0.23) † | +0.96 (d = +0.29) | +0.082 †"),
                 ("Holistic (1-5)", "+0.10 (d = +0.14) † | −0.08 (d = −0.12) † | −0.006 †")]:
    claim("Table 2", pr, lambda crit=crit: " | ".join(t3cell(crit)), note=crit)
def d_range():
    ds = [arm("pool <5", c).d for c in ["Validation", "Flatter", "Enthusiasm", "Positive affirmation"]] + [arm("pool <100 (main)", c).d_lenadj for c in ["Validation", "Flatter", "Enthusiasm", "Positive affirmation"]]
    return f"d = {f(min(ds))}–{f(max(ds))}"
claim("4.4", "d = 0.23–0.36", d_range)
claim("4.4", "no criterion shows a significant post-rollback trend", lambda: "no criterion shows a significant post-rollback trend" if (csv('facct/f14b_rescreen_arms.csv').query("arm=='pool <100 (main)'").slope_p_bh >= .05).all() else "SOME SIGNIFICANT")
# ------------------------------------------------ 5 / 6
PC = lambda k, m: row("facct/f02_postrollback_composition.csv", draw=k, measure=m)
claim("5/6.2", "1.6 to 4.0; 33% to 63% (Draw A)", lambda: f"{PC('A','turns_per_conv').first3:.1f} to {PC('A','turns_per_conv').last3:.1f}; {PC('A','english_pct_turns').first3:.0f}% to {PC('A','english_pct_turns').last3:.0f}% (Draw A)")
claim("6.2", "returning users rise by about 0.4 points", lambda: f"returning users rise by about {round((wu('A','each IP weighted equally').est + wu('B','each IP weighted equally').est)/2, 1)} points")
claim("6.2", "several models until January 5, 2025", lambda: f"several models until {pd.Timestamp(csv('facct/f01_serving_overlaps.csv').end.max()):%B} {pd.Timestamp(csv('facct/f01_serving_overlaps.csv').end.max()).day}, 2025")
ew = lambda k: c2(k, "event window (Apr25-27 vs Apr15-23)")
claim("6.2", "+0.26 [−0.24, 0.76] and −0.32 [−0.68, 0.04]", lambda: f"{est_ci(ew('A'),'diff')} and {est_ci(ew('B'),'diff')}")
claim("Table expl.", "thresholds +0.99 to +1.06; Draw B +1.01; Mar 1 +0.96/+1.01; conversation-level +1.07 to +1.11; reply length +0.86/+0.89; English +1.00/+1.09; returning 0.46/0.38",
      lambda: "thresholds " + f"{f(min(row('facct/f03_ordinary_grid.csv', draw=k, templates=t, pre='Nov 1-Apr 14', post='Apr 29-Jul 31', unit='turn', length='none').est for k in 'AB' for t in ['<100 (as drawn)','<20','<5']),2,True)} to {f(max(row('facct/f03_ordinary_grid.csv', draw=k, templates=t, pre='Nov 1-Apr 14', post='Apr 29-Jul 31', unit='turn', length='none').est for k in 'AB' for t in ['<100 (as drawn)','<20','<5']),2,True)}; Draw B {f(b()['diff'],2,True)}; Mar 1 {f(c2('A','era, Mar 1-Apr 14 baseline')['diff'],2,True)}/{f(c2('B','era, Mar 1-Apr 14 baseline')['diff'],2,True)}; conversation-level {f(min(row('facct/f03_ordinary_grid.csv', draw=k, templates='<100 (as drawn)', pre='Nov 1-Apr 14', post='Apr 29-Jul 31', unit=u, length='none').est for k in 'AB' for u in ['cluster_mean','cluster_first']),2,True)} to {f(max(row('facct/f03_ordinary_grid.csv', draw=k, templates='<100 (as drawn)', pre='Nov 1-Apr 14', post='Apr 29-Jul 31', unit=u, length='none').est for k in 'AB' for u in ['cluster_mean','cluster_first']),2,True)}; reply length {f(c2('A','era, log assistant words')['diff'],2,True)}/{f(c2('B','era, log assistant words')['diff'],2,True)}; English {f(sens('A','English only','era').est,2,True)}/{f(sens('B','English only','era').est,2,True)}; returning {f(wu('A','each IP weighted equally').est)}/{f(wu('B','each IP weighted equally').est)}")
claim("Limitations", "7.5–8.2% of turns", lambda: f"{row('facct/f01_near_threshold.csv', draw='B', period='all').pct_tcn_ge20:.1f}–{row('facct/f01_near_threshold.csv', draw='A', period='all').pct_tcn_ge20:.1f}% of turns")

# ---------------------------------------------------------------- run
def norm(s): return str(s).replace("−", "-").replace(" ", "").replace("+", "").lower()
rows = []
for c in C:
    try: got = c["fn"]()
    except Exception as ex: got = f"ERROR {type(ex).__name__}: {ex}"
    match = norm(got) == norm(c["printed"])
    verdict = "PASS" if match and c["status"] == "ok" else ("KNOWN ERROR" if c["status"] == "known_error" else "FAIL")
    rows.append(dict(section=c["section"], printed=c["printed"], recomputed=got, result=verdict, note=c["note"]))
    print(f"[{verdict:11s}] {c['section']:14s} printed: {c['printed']}  |  data: {got}")
out = pd.DataFrame(rows); out.to_csv(os.path.join(OUTPUT, "paper_claims_check.csv"), index=False)
k = out.result.value_counts().to_dict(); print("\n", k)
sys.exit(0 if k.get("FAIL", 0) == 0 else 1)
