"""十倍股检测引擎：逐日起点 × 5年窗口 × 复权涨幅 × 市值门槛 × 伪象过滤。
口径对齐原文：
- 每个交易日为潜在起点（2005-01-01 ~ 2020-12-31）
- 前看5个日历年，前复权收盘最大涨幅 ≥ threshold
- 起点流通市值 ≥ 10亿美元（当年USDCNY年均汇率折算）
- 伪象过滤：|日收益|>60% 的复权异常起点剔除；起点后60日日均成交额 ≥ 500万元
- 守住检验：首次触及阈值后36个月，复权收盘仍 ≥ 阈值×起点价
- 右删失：触及+36月超出数据末端 → hold=None（不可检验）
"""
import os, json, sys
import numpy as np
import pandas as pd

W = "/mnt/agents/work/tenbagger"
USDCNY = {2005:8.19, 2006:7.97, 2007:7.60, 2008:6.95, 2009:6.83, 2010:6.77,
          2011:6.46, 2012:6.31, 2013:6.20, 2014:6.14, 2015:6.23, 2016:6.64,
          2017:6.75, 2018:6.62, 2019:6.91, 2020:6.90}
MCAP_USD_MIN = 1e9
START_MIN, START_MAX = "2005-01-01", "2020-12-31"
FWD_YEARS = 5
HOLD_MONTHS = 36
LIQ_MIN_WAN = 500.0

_cache = {}

def load(code, kind):
    key = (code, kind)
    if key in _cache: return _cache[key]
    p = f"{W}/{kind}/{code}.pkl"
    df = pd.read_pickle(p) if os.path.exists(p) else None
    _cache[key] = df
    return df

def analyze_stock(code, thr=10.0):
    px = load(code, "data_qfq")
    if px is None or len(px) < 60: return None
    mc = load(code, "data_mcap")
    px = px.dropna(subset=["close"]).reset_index(drop=True)
    dates = pd.to_datetime(px["date"])
    close = px["close"].to_numpy(float)
    n = len(close)
    ret = np.full(n, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    bad_day = np.abs(ret) > 0.60

    smin, smax = pd.Timestamp(START_MIN), pd.Timestamp(START_MAX)
    in_range = ((dates >= smin) & (dates <= smax)).to_numpy()
    idx_starts = np.where(in_range)[0]
    if len(idx_starts) == 0: return None

    if mc is not None and len(mc):
        mcm = mc.set_index(pd.to_datetime(mc["date"]))
        yrs = dates.year.map(USDCNY).to_numpy(float)
        mcap_usd = mcm["float_mcap"].reindex(dates).to_numpy(float) / yrs
        amt = mcm["amount_wan"].reindex(dates).to_numpy(float)
    else:
        mcap_usd = np.full(n, np.nan); amt = np.full(n, np.nan)

    # 分母：合格起点日（市值达标）数
    denom = int(np.nansum(mcap_usd[in_range] >= MCAP_USD_MIN))

    qualified = np.zeros(len(idx_starts), bool)
    for k, i in enumerate(idx_starts):
        if bad_day[i]: continue
        if not np.isnan(mcap_usd[i]) and mcap_usd[i] < MCAP_USD_MIN: continue
        j_end = min(i + 61, n)
        liq = np.nanmean(amt[i:j_end])
        if not np.isnan(liq) and liq < LIQ_MIN_WAN: continue
        end_date = dates.iloc[i] + pd.DateOffset(years=FWD_YEARS)
        j = min(np.searchsorted(dates, end_date, side="right"), n)
        if j <= i + 1: continue
        if close[i:j].max() / close[i] >= thr: qualified[k] = True
    nq = int(qualified.sum())
    if nq == 0:
        return {"code": code, "n_qual": 0, "denom": denom}
    qi = idx_starts[qualified]
    qd = dates.iloc[qi]
    first_q, last_q = qd.iloc[0], qd.iloc[-1]
    i0 = qi[0]
    target = close[i0] * thr
    hits = np.where(close[i0:] >= target)[0]
    touch_idx = i0 + int(hits[0]) if len(hits) else None
    days_to_touch = int(touch_idx - i0) if touch_idx is not None else None
    max_dd = None
    if touch_idx is not None and touch_idx > i0:
        path = close[i0:touch_idx + 1]
        max_dd = float((path / np.maximum.accumulate(path) - 1).min())
    hold = None
    if touch_idx is not None:
        hold_date = dates.iloc[touch_idx] + pd.DateOffset(months=HOLD_MONTHS)
        jh = int(np.searchsorted(dates, hold_date, side="left"))
        if jh < n:
            hold = bool(close[jh] >= target)
    return {"code": code, "n_qual": nq,
            "first_q": str(first_q.date()), "last_q": str(last_q.date()),
            "touch_date": str(dates.iloc[touch_idx].date()) if touch_idx is not None else None,
            "days_to_touch": days_to_touch,
            "years_to_touch": round(days_to_touch / 244, 2) if days_to_touch is not None else None,
            "max_drawdown": max_dd, "hold": hold, "denom": denom}

def run_universe(codes, thr=10.0, out=None, log_every=500):
    rows = []; denom_total = 0
    for k, c in enumerate(codes, 1):
        try:
            r = analyze_stock(c, thr)
        except Exception:
            r = None
        if r:
            denom_total += r.get("denom", 0)
            if r["n_qual"] > 0: rows.append(r)
        if k % log_every == 0:
            print(f"  {k}/{len(codes)} candidates={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    if out: df.to_csv(out, index=False)
    return df, denom_total

if __name__ == "__main__":
    codes = json.load(open(f"{W}/codes.json"))
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out = sys.argv[2] if len(sys.argv) > 2 else f"{W}/results/tenb_{thr:g}x.csv"
    df, denom = run_universe(codes, thr, out)
    qd = int(df["n_qual"].sum()) if len(df) else 0
    print(f"THR={thr:g}x companies={len(df)} qualified_days={qd} denom_days={denom}")
    if denom: print(f"base_rate={qd/denom*100:.3f}%")
