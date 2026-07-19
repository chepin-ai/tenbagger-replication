"""事前可识别性模型：起点日可得特征 → 是否五年十倍（logistic）。
样本：每只股票每月首个交易日（2005-2020），市值≥10亿美元。
特征：log市值、log成交额(60日均)、过去252日收益、过去252日波动、距52周高点回撤、
      市场状态（沪深300距其252周高点回撤）、年份固定效应(简化: 牛熊状态)。
评估：AUC、系数/OR、top10%预测组的基率提升倍数。
"""
import os, json
import numpy as np
import pandas as pd

W = "/mnt/agents/work/tenbagger"
USDCNY = {2005:8.19,2006:7.97,2007:7.60,2008:6.95,2009:6.83,2010:6.77,
          2011:6.46,2012:6.31,2013:6.20,2014:6.14,2015:6.23,2016:6.64,
          2017:6.75,2018:6.62,2019:6.91,2020:6.90}

def build_dataset(max_stocks=None):
    idx = pd.read_pickle(f"{W}/data_index/hs300.pkl")
    idx_d = pd.DatetimeIndex(pd.to_datetime(idx["date"]))
    idx_c = pd.to_numeric(idx["close"], errors="coerce").to_numpy(float)
    idx_peak = pd.Series(idx_c).rolling(250*5, min_periods=60).max().to_numpy()
    idx_dd = idx_c / idx_peak - 1
    idx_map = pd.Series(idx_dd, index=idx_d)

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
        amt = mcm["amount_wan"].reindex(dates).to_numpy(float)
        amt60 = pd.Series(amt).rolling(60, min_periods=20).mean().to_numpy()
        ret = pd.Series(close).pct_change().to_numpy()
        ret252 = close / pd.Series(close).shift(252).to_numpy() - 1
        vol252 = pd.Series(ret).rolling(252, min_periods=120).std().to_numpy() * np.sqrt(244)
        peak52 = pd.Series(close).rolling(244, min_periods=60).max().to_numpy()
        dd52 = close / peak52 - 1
        mdd = idx_map.reindex(dates).to_numpy(float)
        # 每月首个交易日
        ym = pd.Series(dates.year * 100 + dates.month)
        first_of_month = (~ym.duplicated()).to_numpy()
        fwd_rights = np.searchsorted(dates.to_numpy(),
                        dates.to_numpy() + np.timedelta64(int(5*365.25), "D"), side="right")
        for i in range(n):
            if not first_of_month[i]: continue
            if dates[i] < pd.Timestamp("2005-06-01") or dates[i] > pd.Timestamp("2020-12-31"): continue
            if not np.isfinite(mcap[i]) or mcap[i] < 1e9: continue
            j = min(fwd_rights[i], n)
            if j <= i + 200: continue
            y = int(close[i:j].max() / close[i] >= 10)
            rows.append({
                "code": c, "date": str(dates[i].date()), "y": y,
                "log_mcap": np.log(mcap[i]),
                "log_amt": np.log(amt60[i]) if np.isfinite(amt60[i]) and amt60[i] > 0 else np.nan,
                "ret252": ret252[i], "vol252": vol252[i], "dd52": dd52[i], "mkt_dd": mdd[i],
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{W}/results/identify_dataset.csv", index=False)
    return df

def main():
    df = build_dataset()
    print(f"样本: {len(df)} 公司×月, 正例 {df['y'].sum()} ({df['y'].mean()*100:.2f}%)")
    df = df.dropna()
    import statsmodels.api as sm
    X = df[["log_mcap","log_amt","ret252","vol252","dd52","mkt_dd"]]
    X = sm.add_constant(X)
    m = sm.Logit(df["y"], X).fit(disp=0)
    print(m.summary2().tables[1].round(3).to_string())
    df["p"] = m.predict(X)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(df["y"], df["p"])
    top = df[df["p"] >= df["p"].quantile(0.9)]
    lift = top["y"].mean() / df["y"].mean()
    print(f"\nAUC={auc:.3f} | top10%组十倍率={top['y'].mean()*100:.2f}% vs 基率={df['y'].mean()*100:.2f}% | 提升={lift:.1f}倍")
    with open(f"{W}/results/identify_report.txt", "w") as f:
        f.write(m.summary2().tables[1].round(4).to_string())
        f.write(f"\nAUC={auc:.3f} top10%={top['y'].mean()*100:.2f}% base={df['y'].mean()*100:.2f}% lift={lift:.1f}x\n")

if __name__ == "__main__":
    main()
