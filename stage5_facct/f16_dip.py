"""Changepoint segments on the ordinary draws and the post-rollback dip (27B)."""
from fcommon import *
SEG = {"pre (Nov 1-Apr 14)": ("2024-11-01","2025-04-15"), "Apr 15-27": ("2025-04-15","2025-04-28"),
       "Apr 29-May 19": ("2025-04-29","2025-05-20"), "May 20-Jul 31": ("2025-05-20","2025-08-01")}
rows=[]; con=[]
for k in "AB":
    d = draw(k).assign(date=lambda x: x.ts.dt.normalize())
    S = {n: win(d, w) for n, w in SEG.items()}
    for n, x in S.items():
        conv = x.groupby("conv_id").agg(tc=("turn_count","first"), lang=("language","first"))
        rows.append(dict(draw=k, segment=n, turns=len(x), clusters=x.cluster_id.nunique(), mean=x.validation.mean(),
                         english_pct_turns=100*(x.language=="English").mean(), turns_per_conv=conv.tc.mean(), median_aw=x.aw.median()))
    for a,b in [("Apr 15-27","Apr 29-May 19"),("pre (Nov 1-Apr 14)","Apr 29-May 19"),("Apr 29-May 19","May 20-Jul 31")]:
        r = points_contrast(S[a], S[b]); ra = points_contrast(S[a], S[b], covars=["la"])
        con.append(dict(draw=k, a=a, b=b, diff=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p"], d=r["d"], diff_len=ra["diff"], lo_len=ra["lo"], hi_len=ra["hi"], p_len=ra["p"]))
R=pd.DataFrame(rows); C=pd.DataFrame(con); writef(R,"f16_dip_segments.csv"); writef(C,"f16_dip_contrasts.csv")
print(R.round(2).to_string()); print(C.round(3).to_string())
