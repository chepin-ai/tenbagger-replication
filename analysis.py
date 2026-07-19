"""复现结果统计：与原文逐项对表。"""
import os, json, sys
import numpy as np
import pandas as pd

W = "/mnt/agents/work/tenbagger"

def load_results():
    df = pd.read_csv(f"{W}/results/tenb_10x.csv")
    uni = pd.read_json(f"{W}/universe.json")
    df = df.merge(uni, on="code", how="left")
    return df

def main():
    df = load_results()
    print("=== A股十倍股复现 vs 原文 ===")
    n = len(df)
    print(f"十倍股公司数: {n}  (原文 146)")
    qd = df["n_qual"].sum()
    print(f"合格买入日总数: {qd}  (原文 2.3万)")
    print(f"窗口宽度中位数: {df['n_qual'].median():.0f} 个交易日  (原文 A股 80)")
    print(f"窗口≤20天的公司: {(df['n_qual']<=20).sum()}  (原文约101家三市场合计)")
    print(f"窗口≤5天: {(df['n_qual']<=5).sum()}; >500天: {(df['n_qual']>500).sum()}")
    # 守住率
    testable = df.dropna(subset=["hold"])
    hold = testable["hold"].mean()
    print(f"\n守住率(触及后36月仍≥10倍): {hold*100:.1f}% ({int(testable['hold'].sum())}/{len(testable)})  (原文 A股 9%)")
    # 速度梯度
    print("\n=== 速度 vs 守住率 ===")
    for lo, hi, lab in [(0,2,"<2年"),(2,3,"2-3年"),(3,4,"3-4年"),(4,5.01,"4-5年")]:
        sub = testable[(testable["years_to_touch"]>=lo)&(testable["years_to_touch"]<hi)]
        if len(sub):
            print(f"{lab}: {sub['hold'].mean()*100:.1f}% ({int(sub['hold'].sum())}/{len(sub)})")
    print("(原文: <3年合计 3.5% | 3-4年 17% | 4-5年 28%)")
    # 回撤
    dd = df["max_drawdown"].dropna()
    print(f"\n=== 途中最大回撤 ===")
    print(f"中位数: {dd.median()*100:.1f}%  (原文 45-47%)")
    print(f"腰斩(≤-50%)比例: {(dd<=-0.5).mean()*100:.1f}%  (原文 四成)")
    # 月度聚集
    print("\n=== 上车月分布 Top10 ===")
    fq = pd.to_datetime(df["first_q"])
    print(fq.dt.to_period("M").value_counts().head(10))
    # 具名个案
    print("\n=== 具名个案 ===")
    for code, name, art in [("600887","伊利","32天 2008.12-2009.2"),("300750","宁德时代","54天")]:
        r = df[df["code"]==code]
        if len(r):
            r=r.iloc[0]
            print(f"{name}: 合格买入日={r['n_qual']} 窗口 {r['first_q']}->{r['last_q']}  (原文: {art})")
        else:
            print(f"{name}: 未入选 (原文: {art})")

if __name__ == "__main__":
    main()
