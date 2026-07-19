"""港股十倍股检测（基于已抓取的腾讯hfq港股数据）。"""
import os, sys
import numpy as np
import pandas as pd

W = "/mnt/agents/work/tenbagger"
START_MIN, START_MAX = "2005-01-01", "2020-12-31"

def analyze(px, thr=10.0):
    px = px[px["close"] > 0].reset_index(drop=True)
    if len(px) < 120: return None
    dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
    close = px["close"].to_numpy(float)
    n = len(close)
    rights = np.searchsorted(dates.to_numpy(),
                dates.to_numpy() + np.timedelta64(int(5*365.25), "D"), side="right")
    in_range = ((dates >= START_MIN) & (dates <= START_MAX))
    idx = np.where(in_range)[0]
    qual = []
    from collections import deque
    dq = deque(); j = 0
    fwd_max = np.zeros(n)
    for i in range(n):
        r = min(rights[i], n)
        while j < r:
            while dq and close[dq[-1]] <= close[j]: dq.pop()
            dq.append(j); j += 1
        while dq and dq[0] < i: dq.popleft()
        fwd_max[i] = close[dq[0]] if dq else close[i]
    ratio = fwd_max / close
    qi = idx[ratio[idx] >= thr]
    if len(qi) == 0: return None
    i0 = qi[0]; target = close[i0] * thr
    hits = np.where(close[i0:] >= target)[0]
    t_i = i0 + int(hits[0])
    jh = int(np.searchsorted(dates, dates[t_i] + pd.DateOffset(months=36)))
    hold = bool(close[jh] >= target) if jh < n else None
    return {"n_qual": len(qi), "first_q": str(dates[qi[0]].date()),
            "last_q": str(dates[qi[-1]].date()),
            "years_to_touch": round((t_i - i0) / 244, 2), "hold": hold}

def main():
    out = f"{W}/results/tenb_hk.csv"
    rows = []
    files = [f for f in os.listdir(f"{W}/data_hk") if f.endswith(".pkl")]
    for f in files:
        sym = f[:-4]
        try:
            px = pd.read_pickle(f"{W}/data_hk/{f}")
            r = analyze(px)
            if r: rows.append({"sym": sym, **r})
        except Exception: pass
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"HK stocks scanned: {len(files)}, tenbaggers: {len(df)} (无市值门槛)")
    if len(df):
        print(df.sort_values("n_qual", ascending=False).head(15).to_string(index=False))

if __name__ == "__main__":
    main()
