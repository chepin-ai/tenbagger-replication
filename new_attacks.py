"""新方向攻击包：
A. 十倍友好度指数（逐月基率 vs 市场状态）——When>What 的可测化
B. 左尾镜像（5年-90%基率 vs +10倍基率）——偏态对称性破缺
C. 十倍²（A股百倍股）——与Mayer谱系对接
D. 月度聚集重尾性（雪崩动力学检验）
E. EVT/GPD尾部拟合——基率的极值理论重构
"""
import os, json
import numpy as np
import pandas as pd
from collections import deque

W = "/mnt/agents/work/tenbagger"
USDCNY = {2005:8.19,2006:7.97,2007:7.60,2008:6.95,2009:6.83,2010:6.77,
          2011:6.46,2012:6.31,2013:6.20,2014:6.14,2015:6.23,2016:6.64,
          2017:6.75,2018:6.62,2019:6.91,2020:6.90}

def monthly_stats():
    """每个股票×月首起点：fwd_max, fwd_min, mcap, 日期。"""
    out_csv = f"{W}/results/monthly_stats.csv"
    codes = json.load(open(f"{W}/codes_main.json"))
    rows = []
    for c in codes:
        ph, pm = f"{W}/data_hfq/{c}.pkl", f"{W}/data_mcap/{c}.pkl"
        if not (os.path.exists(ph) and os.path.exists(pm)): continue
        try:
            px = pd.read_pickle(ph); mc = pd.read_pickle(pm)
        except Exception: continue
        px = px[px["close"] > 0].reset_index(drop=True)
        if len(px) < 320: continue
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        close = px["close"].to_numpy(float)
        n = len(close)
        mcm = mc.set_index(pd.to_datetime(mc["date"]))
        yrs = pd.Series(dates.year).map(USDCNY).to_numpy(float)
        mcap = mcm["float_mcap"].reindex(dates).to_numpy(float) / yrs
        ym = pd.Series(dates.year * 100 + dates.month)
        first_of_month = (~ym.duplicated()).to_numpy()
        rights = np.searchsorted(dates.to_numpy(),
                    dates.to_numpy() + np.timedelta64(int(5*365.25), "D"), side="right")
        dq_max, dq_min = deque(), deque(); j = 0
        fmax = np.zeros(n); fmin = np.zeros(n)
        for i in range(n):
            r = min(rights[i], n)
            while j < r:
                while dq_max and close[dq_max[-1]] <= close[j]: dq_max.pop()
                while dq_min and close[dq_min[-1]] >= close[j]: dq_min.pop()
                dq_max.append(j); dq_min.append(j); j += 1
            while dq_max and dq_max[0] < i: dq_max.popleft()
            while dq_min and dq_min[0] < i: dq_min.popleft()
            fmax[i] = close[dq_max[0]]; fmin[i] = close[dq_min[0]]
        for i in range(n):
            if not first_of_month[i]: continue
            if dates[i] < pd.Timestamp("2005-06-01") or dates[i] > pd.Timestamp("2020-12-31"): continue
            if not np.isfinite(mcap[i]) or mcap[i] < 1e9: continue
            if min(rights[i], n) <= i + 200: continue
            rows.append({"code": c, "month": int(ym.iloc[i]),
                         "rmax": fmax[i]/close[i], "rmin": fmin[i]/close[i]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def hundred_baggers():
    """严格十倍股样本：触及日后再5年是否再十倍（=百倍）。"""
    fin = pd.read_csv(f"{W}/results/tenb_10x_strict.csv")
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    hb = []
    for _, r in fin.iterrows():
        if pd.isna(r["touch_date"]): continue
        p = f"{W}/data_hfq/{r['code']}.pkl"
        if not os.path.exists(p): continue
        px = pd.read_pickle(p); px = px[px["close"] > 0].reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        close = px["close"].to_numpy(float)
        t = int(np.searchsorted(dates, pd.Timestamp(r["touch_date"])))
        if t >= len(close) - 10: continue
        j = min(int(np.searchsorted(dates, dates[t] + pd.DateOffset(years=5))), len(close))
        ratio2 = close[t:j].max() / close[t] if j > t + 1 else 0
        if ratio2 >= 10: hb.append(r["code"])
    return hb

def main():
    df = monthly_stats()
    print(f"[stats] {len(df)} 公司×月样本")
    # A. 十倍友好度指数
    g = df.groupby("month").agg(n=("rmax","size"),
                                tenb=("rmax", lambda s: (s>=10).mean()),
                                crash=("rmin", lambda s: (s<=0.1).mean()))
    g = g[g["n"] >= 50]
    print("\n[A] 十倍友好度指数（逐月基率, 概率范围）:")
    print(f"  最高: {g['tenb'].idxmax()} {g['tenb'].max()*100:.2f}% | 最低非零: {g[g['tenb']>0]['tenb'].min()*100:.3f}%")
    print(f"  基率=0的月份占比: {(g['tenb']==0).mean()*100:.0f}%")
    top10 = g["tenb"].nlargest(10)
    print("  友好度Top10月:", [(int(m), f"{v*100:.1f}%") for m,v in top10.items()])
    # B. 左尾镜像
    p_up = (df["rmax"]>=10).mean(); p_dn = (df["rmin"]<=0.1).mean()
    print(f"\n[B] 右尾(+10x)基率: {p_up*100:.3f}% | 左尾(-90%)基率: {p_dn*100:.3f}% | 不对称比: {p_up/max(p_dn,1e-9):.2f}")
    q = df.groupby(df["month"]//100)[["rmax","rmin"]].agg(
        up=("rmax", lambda s:(s>=10).mean()), dn=("rmin", lambda s:(s<=0.1).mean()))
    print("  分年度不对称:", {int(y): f"{r['up']/max(r['dn'],1e-9):.1f}" for y,r in q.iterrows()})
    # C. 十倍²
    hb = hundred_baggers()
    print(f"\n[C] A股百倍股(十倍后5年内再十倍): {len(hb)} 家 → {hb}")
    # D. 月度聚集重尾
    monthly_new = df[df["rmax"]>=10].groupby("month").size()
    counts = monthly_new.values
    cc = np.sort(counts)[::-1]
    ccdf = np.arange(1, len(cc)+1)/len(cc)
    mask = cc >= np.percentile(cc, 50)
    if mask.sum() > 3:
        slope = np.polyfit(np.log(cc[mask]), np.log(ccdf[mask]), 1)[0]
        print(f"\n[D] 月度十倍起点计数: max={counts.max()}, 中位={np.median(counts):.0f}, CCDF幂律斜率≈{slope:.2f}")
    print(f"  Top5月占比: {np.sort(counts)[-5:].sum()/counts.sum()*100:.0f}% (共{len(counts)}个月)")
    # E. EVT/GPD
    try:
        from scipy import stats as ss
        exc = df["rmax"][df["rmax"]>3] - 3
        shape, loc, scale = ss.genpareto.fit(exc, floc=0)
        p3 = (df["rmax"]>3).mean()
        p10_evt = p3 * (1 - ss.genpareto.cdf(10-3, shape, loc, scale))
        p10_emp = (df["rmax"]>=10).mean()
        print(f"\n[E] GPD拟合(u=3x): ξ={shape:.3f}, β={scale:.2f}")
        print(f"  P(≥10x) 经验={p10_emp*100:.3f}% vs EVT外推={p10_evt*100:.3f}%")
    except Exception as e:
        print("[E] GPD fit failed:", e)
    g.to_csv(f"{W}/results/friendliness_index.csv")

if __name__ == "__main__":
    main()
