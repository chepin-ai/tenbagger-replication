"""十倍股检测引擎 v2：滑动窗口O(n)向量化 + 增量落盘（抗沙箱重置）。"""
import os, json, sys
import numpy as np
import pandas as pd
from collections import deque

W = "/mnt/agents/work/tenbagger"
USDCNY = {2005:8.19, 2006:7.97, 2007:7.60, 2008:6.95, 2009:6.83, 2010:6.77,
          2011:6.46, 2012:6.31, 2013:6.20, 2014:6.14, 2015:6.23, 2016:6.64,
          2017:6.75, 2018:6.62, 2019:6.91, 2020:6.90}
MCAP_USD_MIN = 1e9
START_MIN, START_MAX = "2005-01-01", "2020-12-31"
FWD_YEARS = 5
HOLD_MONTHS = 36
LIQ_MIN_WAN = 500.0

def load(code, kind):
    p = f"{W}/{kind}/{code}.pkl"
    return pd.read_pickle(p) if os.path.exists(p) else None

def forward_window_max(dates_np, close, years=FWD_YEARS):
    """fwd_max[i] = max(close[i : j])，j 为 dates[i]+years 之后的位置。双指针+单调栈 O(n)。"""
    n = len(close)
    fwd_max = np.zeros(n)
    touch = np.full(n, -1, dtype=np.int64)  # 首次 >= thr*close[i] 的位置（在窗口内）
    dq = deque()  # 单调递减栈，存索引
    j = 0
    # 预先计算每个 i 的窗口右端
    yrs = dates_np.astype("datetime64[Y]").astype(int) + 1970
    end_dates = dates_np + np.array([years]*n) * np.timedelta64(365, "D") \
                + (np.isin(yrs, []) )  # 近似：按365.25天×年
    # 精确：用 DateOffset 太重；用 1826 天近似5年（含闰年）
    end_dates = dates_np + np.timedelta64(int(years*365.25), "D")
    rights = np.searchsorted(dates_np, end_dates, side="right")
    for i in range(n):
        r = min(rights[i], n)
        while j < r:
            while dq and close[dq[-1]] <= close[j]: dq.pop()
            dq.append(j); j += 1
        while dq and dq[0] < i: dq.popleft()
        fwd_max[i] = close[dq[0]] if dq else close[i]
    return fwd_max

def analyze_stock(code, thr=10.0):
    px = load(code, "data_hfq")
    if px is None or len(px) < 60: return None
    px = px.dropna(subset=["close"]).reset_index(drop=True)
    if (px["close"] <= 0).any():
        px = px[px["close"] > 0].reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
    dates_np = dates.to_numpy()
    close = px["close"].to_numpy(float)
    n = len(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.full(n, np.nan); ret[1:] = close[1:] / close[:-1] - 1
    bad_day = np.abs(ret) > 0.60

    smin, smax = np.datetime64(START_MIN), np.datetime64(START_MAX)
    in_range = (dates_np >= smin) & (dates_np <= smax)
    idx_starts = np.where(in_range)[0]
    if len(idx_starts) == 0: return None

    mc = load(code, "data_mcap")
    if mc is not None and len(mc):
        mcm = mc.set_index(pd.to_datetime(mc["date"]))
        yrs = pd.Series(dates.year).map(USDCNY).to_numpy(float)
        mcap_usd = mcm["float_mcap"].reindex(dates).to_numpy(float) / yrs
        amt = mcm["amount_wan"].reindex(dates).to_numpy(float)
    else:
        mcap_usd = np.full(n, np.nan); amt = np.full(n, np.nan)
    denom = int(np.nansum(mcap_usd[in_range] >= MCAP_USD_MIN))

    # 流动性：60日滚动日均成交额
    amt_s = pd.Series(amt).rolling(60, min_periods=20).mean().shift(-60).to_numpy()

    fwd_max = forward_window_max(dates_np, close)
    ratio = fwd_max / close
    qual = np.zeros(n, bool)
    st = idx_starts
    ok = ~bad_day[st]
    if os.environ.get("STRICT_MCAP") == "1":
        ok &= ~np.isnan(mcap_usd[st]) & (mcap_usd[st] >= MCAP_USD_MIN)
    else:
        ok &= ~(~np.isnan(mcap_usd[st]) & (mcap_usd[st] < MCAP_USD_MIN))
    ok &= ~(~np.isnan(amt_s[st]) & (amt_s[st] < LIQ_MIN_WAN))
    qual[st[ok & (ratio[st] >= thr)]] = True
    nq = int(qual.sum())
    if nq == 0:
        return {"code": code, "n_qual": 0, "denom": denom}
    qi = np.where(qual)[0]
    first_q, last_q = dates[qi[0]], dates[qi[-1]]
    i0 = qi[0]; target = close[i0] * thr
    hits = np.where(close[i0:] >= target)[0]
    touch_idx = i0 + int(hits[0]) if len(hits) else None
    days_to_touch = int(touch_idx - i0) if touch_idx is not None else None
    max_dd = None
    if touch_idx is not None and touch_idx > i0:
        path = close[i0:touch_idx + 1]
        max_dd = float((path / np.maximum.accumulate(path) - 1).min())
    hold = None
    if touch_idx is not None:
        hold_date = dates[touch_idx] + pd.DateOffset(months=HOLD_MONTHS)
        jh = int(np.searchsorted(dates, hold_date))
        if jh < n: hold = bool(close[jh] >= target)
    return {"code": code, "n_qual": nq,
            "first_q": str(first_q.date()), "last_q": str(last_q.date()),
            "touch_date": str(dates[touch_idx].date()) if touch_idx is not None else None,
            "days_to_touch": days_to_touch,
            "years_to_touch": round(days_to_touch / 244, 2) if days_to_touch is not None else None,
            "max_drawdown": max_dd, "hold": hold, "denom": denom,
            "has_mcap": bool(mc is not None and len(mc))}

if __name__ == "__main__":
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out = sys.argv[2] if len(sys.argv) > 2 else f"{W}/results/tenb_{thr:g}x.csv"
    codes = json.load(open(f"{W}/codes.json"))
    rows, denom_total = [], 0
    for k, c in enumerate(codes, 1):
        try:
            r = analyze_stock(c, thr)
        except Exception as e:
            r = None
        if r:
            denom_total += r.get("denom", 0)
            if r["n_qual"] > 0: rows.append(r)
        if k % 300 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)  # 增量落盘
            print(f"  {k}/{len(codes)} candidates={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    qd = int(df["n_qual"].sum()) if len(df) else 0
    print(f"THR={thr:g}x companies={len(df)} qualified_days={qd} denom_days={denom_total}")
    if denom_total: print(f"base_rate={qd/denom_total*100:.3f}%")
