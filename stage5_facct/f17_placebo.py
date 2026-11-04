"""Placebo: how often does a 13-day window followed (after a 1-day gap) by a 21-day window, placed anywhere
in the pre-period, show a drop as large as the Apr 15-27 -> Apr 29-May 19 dip? Also same contrast on English only."""
from fcommon import *
rows = []
for k in "AB":
    d = draw(k).assign(date=lambda x: x.ts.dt.normalize())
    obs = win(d,("2025-04-29","2025-05-20")).validation.mean() - win(d,("2025-04-15","2025-04-28")).validation.mean()
    pre = d[d.date < "2025-04-15"]; daily = pre.groupby("date").validation.agg(["sum","size"])
    days = pd.date_range("2024-11-01","2025-04-14"); daily = daily.reindex(days)
    diffs=[]
    for s in days:
        a=daily.loc[s:s+pd.Timedelta(days=12)]; b=daily.loc[s+pd.Timedelta(days=14):s+pd.Timedelta(days=34)]
        if b.index.max() is pd.NaT or s+pd.Timedelta(days=34) > days[-1]: break
        if a["size"].sum()<1000 or b["size"].sum()<1000: continue
        diffs.append(b["sum"].sum()/b["size"].sum()-a["sum"].sum()/a["size"].sum())
    diffs=np.array(diffs)
    en = d[d.language=="English"]
    en_obs = win(en,("2025-04-29","2025-05-20")).validation.mean() - win(en,("2025-04-15","2025-04-28")).validation.mean()
    print(f"Draw {k}: dip {obs:+.2f}; placebo windows {len(diffs)}; |diff|>=|dip| in {np.mean(np.abs(diffs)>=abs(obs))*100:.0f}%; drops <= dip {np.mean(diffs<=obs)*100:.0f}%; placebo range {diffs.min():+.2f} to {diffs.max():+.2f}; English-only dip {en_obs:+.2f}")
    # era level shift vs placebo: same-length halves
    rows.append(dict(draw=k, dip=obs, placebo_windows=len(diffs), pct_abs_ge_dip=100*np.mean(np.abs(diffs) >= abs(obs)),
                     pct_drop_le_dip=100*np.mean(diffs <= obs), placebo_min=diffs.min(), placebo_max=diffs.max(), english_only_dip=en_obs))
writef(pd.DataFrame(rows), "f17_placebo.csv")
