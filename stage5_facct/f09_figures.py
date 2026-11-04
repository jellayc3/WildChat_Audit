"""Main-text figures for the FAccT draft. All data from package frames/outputs."""
from fcommon import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
FIG = os.path.join(FIGURES, "facct"); os.makedirs(FIG, exist_ok=True)
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 300})
UPD, ROLL_END = pd.Timestamp("2025-04-25"), pd.Timestamp("2025-04-29")
def mark(ax, swap=True):
    ax.axvspan(UPD, ROLL_END, color="#f2d7c9", lw=0, zorder=0)
    if swap: ax.axvline(TRANSITION, color=INK2, ls="--", lw=.9, zorder=1)
def fmt(ax, sparse=False):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 1, 3, 5, 7]) if sparse else mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
def wk_cluster_ci(d, col="validation"):
    rows = []
    for w, g in d.groupby(pd.Grouper(key="ts", freq=PELT_RESAMPLE)):
        if len(g) < 2: continue
        m = sm.OLS(g[col].values, np.ones(len(g))).fit(cov_type="cluster", cov_kwds={"groups": g[CLUSTER_UNIT].values})
        rows.append((w, m.params[0], m.params[0] - 1.96 * m.bse[0], m.params[0] + 1.96 * m.bse[0]))
    return pd.DataFrame(rows, columns=["wk", "mean", "lo", "hi"]).set_index("wk")
def gaps(s, maxgap=8):
    """insert NaN where consecutive weekly bins are >1 week apart (coverage gaps)"""
    idx = s.index; out = s.copy()
    br = [idx[i] for i in range(1, len(idx)) if (idx[i] - idx[i-1]).days > maxgap]
    for b in br: out.loc[b - pd.Timedelta(days=1)] = np.nan
    return out.sort_index()

def fig1():
    E = event_scored(); E = E[E.ts < SPAN[1]]
    _, cps = changepoints(E)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
    for ax, name in zip(axs.ravel(), CORE6):
        s = gaps(weekly_means(E, col(name))["mean"])
        mark(ax, swap=False); ax.plot(s.index, s.values, color=C1, lw=1.6)
        for x in cps[name]: ax.axvline(x, color=INK, lw=.8, ls=(0, (2, 2)))
        ax.set_title(name, loc="left"); fmt(ax, sparse=True)
    for ax in axs[:, 0]: ax.set_ylabel("Weekly mean (0–10)")
    fig.tight_layout(); p = os.path.join(FIG, "fig1_naive_weekly_pelt.png"); fig.savefig(p); plt.close(fig); print(p)

def fig2():
    E = event_scored().rename(columns={col("Validation"): "validation"}); E = E[E.ts < SPAN[1]]
    a = gaps(weekly(E)["mean"]); r = gaps(weekly(E[~E.templated])["mean"])
    sh = pd.read_csv(os.path.join(OUT_F, "f01_weekly_templated_share.csv"), parse_dates=["ts"]).set_index("ts").templated_pct
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    mark(ax); ax.plot(a.index, a, color=C1, lw=1.8); ax.plot(r.index, r, color=C2, lw=1.8)
    ax.text(a.dropna().index[-1] + pd.Timedelta(days=4), a.dropna().iloc[-1], "All traffic", va="center", fontsize=8, color=INK)
    ax.text(r.dropna().index[-1] + pd.Timedelta(days=4), r.dropna().iloc[-1], "Templated\nremoved", va="center", fontsize=8, color=INK)
    ax.set_xlim(pd.Timestamp("2024-10-25"), pd.Timestamp("2025-08-28"))
    ax.set_ylabel("Validation, weekly mean"); ax.set_title("(a) Weekly mean validation, event sample (12B judge)", loc="left")
    mark(bx); bx.bar(sh.index, sh.values, width=5, color=INK2); bx.set_ylabel("% of turns"); bx.set_ylim(0, 100)
    bx.set_title("(b) Share of turns in templated conversations", loc="left"); fmt(bx)
    fig.tight_layout(); p = os.path.join(FIG, "fig2_composition.png"); fig.savefig(p); plt.close(fig); print(p)

def fig3():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    mark(ax)
    for k, c in [("A", C1), ("B", C2)]:
        d = draw(k); d = d[d.ts < SPAN[1]]; w = wk_cluster_ci(d)
        w = w.reindex(gaps(w["mean"]).index)
        if k == "A": ax.fill_between(w.index, w.lo, w.hi, color=C1, alpha=.18, lw=0)
        ax.plot(w.index, w["mean"], color=c, lw=1.8 if k == "A" else 1.4, ls="-" if k == "A" else (0, (4, 2)))
        last = w.dropna().iloc[-4]; ax.annotate(f"Draw {k}", (w.dropna().index[-4], last["mean"]), xytext=(0, 9 if k == "A" else -13), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate("serving label changes\ngpt-4o-mini → gpt-4.1-mini", (TRANSITION, 1.75), xytext=(-118, 0), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.annotate("disclosed update\nto rollback", (UPD, 4.45), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_ylabel("Validation, weekly mean (27B judge)"); ax.set_ylim(1.5, 4.8); fmt(ax)
    fig.tight_layout(); p = os.path.join(FIG, "fig3_ordinary_draws.png"); fig.savefig(p); plt.close(fig); print(p)

def fig4():
    T = trend_frame(); W = T.groupby("model").ts.agg(["min", "max", "size"]).sort_values("min")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (m, r) in enumerate(W.iterrows()):
        col_ = INK2 if m in PAIR_EXCLUDE else C1
        ax.barh(i, (r["max"] - r["min"]).days, left=r["min"], height=.55, color=col_)
        ax.text(r["max"] + pd.Timedelta(days=6), i, m, va="center", fontsize=7, color=INK)
    last = pd.Timestamp("2025-01-05")
    ax.axvspan(last, pd.Timestamp("2025-08-01"), color="#efeee9", lw=0, zorder=0)
    ax.axvline(TRANSITION, color=INK2, ls="--", lw=.9)
    ax.text(last - pd.Timedelta(days=6), -1.0, "one label at a time\nfrom Jan 6, 2025", fontsize=7.5, color=INK2, va="top", ha="right")
    ax.text(TRANSITION + pd.Timedelta(days=6), -1.0, "label changes\nApr 15, 2025", fontsize=7.5, color=INK2, va="top")
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_ylim(len(W) - .4, -1.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(pd.Timestamp("2023-03-15"), pd.Timestamp("2026-01-15")); ax.grid(axis="y", visible=False)
    fig.tight_layout(); p = os.path.join(FIG, "fig4_serving_windows.png"); fig.savefig(p); plt.close(fig); print(p)


def figS_daily():
    E = event_scored().rename(columns={col("Validation"): "validation"})
    E = E[(E.ts >= "2025-04-01") & (E.ts < "2025-06-16")].dropna(subset=["validation"])
    rows = []
    for day, g in E.groupby(E.ts.dt.normalize()):
        if len(g) < DAILY_MIN_TURNS: continue
        m = sm.OLS(g.validation.values, np.ones(len(g))).fit(cov_type="cluster", cov_kwds={"groups": g[CLUSTER_UNIT].values})
        rows.append((day, m.params[0], m.params[0] - 1.96 * m.bse[0], m.params[0] + 1.96 * m.bse[0]))
    d = pd.DataFrame(rows, columns=["day", "m", "lo", "hi"]).set_index("day")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max()))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for (a, b), lab in [(W_BASE, "baseline"), (W_EVENT, "event"), (W_POST, "post-rollback")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#efeee9", lw=0, zorder=0)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 8.6, lab, ha="center", fontsize=7.5, color=INK2)
    ax.fill_between(d.index, d.lo, d.hi, color=C1, alpha=.18, lw=0); ax.plot(d.index, d.m, color=C1, lw=1.6)
    ax.set_ylabel("Validation, daily mean (12B)"); ax.set_ylim(0, 9.2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout(); p = os.path.join(FIG, "figS1_daily_validation.png"); fig.savefig(p); plt.close(fig); print(p)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); figS_daily()
