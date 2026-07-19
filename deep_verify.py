"""F命题深度验证 + 开放问题攻击：
P1. 港股基率界（无门槛上界 + 分母全量）
P2. 港股 θ² 事件（百倍股）——F8 跨市场
P3. 港股流动性代理过滤（开放问题a）
P4. EVT 多阈值深验（F6 深）：GPD 对 5x/20x 外推 vs 经验
P5. 动态尾指 ξ(t)：按起点年分组的 GPD 拟合（开放问题b）
"""
import os, json
import numpy as np
import pandas as pd
from collections import deque

W = "/mnt/agents/work/tenbagger"

def hk_base_and_chain():
    files = [f for f in os.listdir(f"{W}/data_hk") if f.endswith(".pkl")]
    cand = pd.read_csv(f"{W}/results/tenb_hk.csv")
    qual_days = int(cand["n_qual"].sum())
    total_days = 0
    chains = []
    for f in files:
        try:
            px = pd.read_pickle(f"{W}/data_hk/{f}")
        except Exception: continue
        px = px[px["close"] > 0].reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        total_days += int(((dates >= "2005-01-01") & (dates <= "2020-12-31")).sum())
    # θ²：对每个候选，触及后再5年
    for _, r in cand.iterrows():
        p = f"{W}/data_hk/{r['sym']}.pkl"
        if not os.path.exists(p): continue
        px = pd.read_pickle(p); px = px[px["close"] > 0].reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        close = px["close"].to_numpy(float)
        # 重新定位触及日
        i0 = int(np.searchsorted(dates, pd.Timestamp(r["first_q"])))
        if i0 >= len(close): continue
        target = close[i0] * 10
        after = close[i0:]
        hits = np.where(after >= target)[0]
        if not len(hits): continue
        t = i0 + int(hits[0])
        j = min(int(np.searchsorted(dates, dates[t] + pd.DateOffset(years=5))), len(close))
        if j > t + 10 and close[t:j].max() / close[t] >= 10:
            chains.append(r["sym"])
    br = qual_days / total_days * 100 if total_days else 0
    print(f"[P1] 港股无门槛基率上界: {qual_days}/{total_days} = {br:.3f}% (候选{len(cand)}家, 原文过滤后40家)")
    print(f"[P2] 港股θ²事件(百倍): {len(chains)} 家 → {chains}")
    return chains

def hk_liquidity_proxy():
    cand = pd.read_csv(f"{W}/results/tenb_hk.csv")
    kept = []
    for _, r in cand.iterrows():
        p = f"{W}/data_hk/{r['sym']}.pkl"
        if not os.path.exists(p): continue
        px = pd.read_pickle(p)
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        i0 = int(np.searchsorted(dates, pd.Timestamp(r["first_q"])))
        vol = pd.to_numeric(px["volume"], errors="coerce").to_numpy(float)
        cls = pd.to_numeric(px["close"], errors="coerce").to_numpy(float)
        if i0 >= len(vol): continue
        w = vol[i0:i0+60] * cls[i0:i0+60]  # 成交额代理（后复权价×量，仅供相对比较）
        avg_amt = np.nanmean(w)
        kept.append((r["sym"], r["n_qual"], avg_amt))
    df = pd.DataFrame(kept, columns=["sym","n_qual","avg_amt"])
    for thr in [1e7, 5e7, 1e8]:
        sub = df[df["avg_amt"] >= thr]
        print(f"[P3] 成交额代理≥{thr/1e7:.0f}百万港元/日: 剩 {len(sub)} 家 (原文40家)")
    return df

def evt_deep():
    df = pd.read_csv(f"{W}/results/monthly_stats.csv")
    from scipy import stats as ss
    r = df["rmax"].to_numpy()
    exc = r[r > 3] - 3
    shape, loc, scale = ss.genpareto.fit(exc, floc=0)
    p3 = (r > 3).mean()
    print(f"[P4] GPD(u=3): ξ={shape:.3f} β={scale:.3f}")
    for x in [5, 10, 15, 20]:
        emp = (r >= x).mean()
        evt = p3 * (1 - ss.genpareto.cdf(x-3, shape, loc, scale))
        print(f"  P(≥{x}x): 经验={emp*100:.3f}% EVT={evt*100:.3f}% 误差={abs(evt-emp)/max(emp,1e-9)*100:.0f}%")
    # P5 动态尾指：按起点年
    df["yr"] = df["month"] // 100
    print("[P5] 动态尾指 ξ(t):")
    for y, g in df.groupby("yr"):
        rr = g["rmax"].to_numpy()
        e = rr[rr > 3] - 3
        if len(e) < 30:
            print(f"  {y}: 样本不足(n={len(e)})"); continue
        s, l, sc = ss.genpareto.fit(e, floc=0)
        print(f"  {y}: ξ={s:.3f} (n={len(e)}, P≥10x={ (rr>=10).mean()*100:.2f}%)")

if __name__ == "__main__":
    hk_base_and_chain()
    hk_liquidity_proxy()
    evt_deep()
