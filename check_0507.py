import pandas as pd, numpy as np, os, json
W="/mnt/agents/work/tenbagger"
USDCNY={2005:8.19,2007:7.60}
codes=json.load(open(f"{W}/codes_main.json"))
def load(p):
    try:
        if os.path.exists(p) and os.path.getsize(p)>100: return pd.read_pickle(p)
    except Exception: return None
    return None
def frac_10x(ym, yr):
    n_eligible=n_qual=0
    for c in codes:
        px=load(f"{W}/data_hfq/{c}.pkl"); mc=load(f"{W}/data_mcap/{c}.pkl")
        if px is None or mc is None or not len(px) or not len(mc): continue
        dates=pd.DatetimeIndex(pd.to_datetime(px["date"]))
        i=int(np.searchsorted(dates, pd.Timestamp(ym)))
        if i>=len(dates) or dates[i]>pd.Timestamp(ym)+pd.Timedelta(days=31): continue
        mcm=mc.set_index(pd.to_datetime(mc["date"]))
        fv=mcm["float_mcap"].reindex(dates).iloc[i]
        if not np.isfinite(fv) or fv/USDCNY[yr]<1e9: continue
        n_eligible+=1
        close=px["close"].to_numpy(float)
        j=min(int(np.searchsorted(dates, dates[i]+pd.DateOffset(years=5))), len(close))
        if j>i+1 and close[i:j].max()/close[i]>=10: n_qual+=1
    return n_eligible,n_qual
out=[]
for ym,yr in [("2005-01-04",2005),("2007-01-04",2007)]:
    e,q=frac_10x(ym,yr)
    out.append(f"{ym[:7]}: eligible={e} tenbagger={q} prob={q/max(e,1)*100:.1f}%")
open(f"{W}/results/check_0507.txt","w").write("\n".join(out))
print("done")
