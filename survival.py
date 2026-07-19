"""生存分析：替代粗守住率。
事件定义：首次触及10倍后，收盘价首次跌破 10×起点价 的日期（失守）。
删失：数据末端仍未失守 → 右删失。
输出：KM生存曲线（全体/按速度组/按成色组）+ Cox比例风险模型。
"""
import os, json
import numpy as np
import pandas as pd

W = "/mnt/agents/work/tenbagger"
THR = 10.0

def build_survival_table():
    fin = pd.read_csv(f"{W}/results/tenb_10x_strict.csv")
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    dec = pd.read_csv(f"{W}/results/eps_decomp.csv")
    dec["code"] = dec["code"].astype(str).str.zfill(6)
    dec = dec.drop_duplicates("code")
    fin = fin.drop_duplicates("code").merge(dec[["code","earn_mult","pe_mult","pe0"]], on="code", how="left")
    rows = []
    for _, r in fin.iterrows():
        if pd.isna(r["touch_date"]): continue
        p = f"{W}/data_hfq/{r['code']}.pkl"
        if not os.path.exists(p): continue
        px = pd.read_pickle(p)
        px = px[px["close"] > 0].reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(px["date"]))
        close = px["close"].to_numpy(float)
        i0 = int(np.searchsorted(dates, pd.Timestamp(r["first_q"])))
        t_i = int(np.searchsorted(dates, pd.Timestamp(r["touch_date"])))
        if i0 >= len(close) or t_i >= len(close): continue
        target = close[i0] * THR
        after = close[t_i:]
        last_date = dates[-1]
        above_idx = np.where(after >= target)[0]
        if len(above_idx) == 0:
            duration, event = 0.1, 1
        else:
            last_above = t_i + int(above_idx[-1])
            remain_days = (last_date - dates[last_above]).days
            if remain_days >= 365 and close[last_above + 1:].max() < target:
                duration = (dates[last_above] - dates[t_i]).days / 30.44
                event = 1
            else:
                duration = (last_date - dates[t_i]).days / 30.44
                event = 0
        rows.append({
            "code": r["code"], "duration_m": duration, "event": event,
            "years_to_touch": r["years_to_touch"],
            "touch_year": pd.Timestamp(r["touch_date"]).year,
            "fast": int(r["years_to_touch"] < 3) if pd.notna(r["years_to_touch"]) else np.nan,
            "earn_dom": int(r["earn_mult"] >= r["pe_mult"]) if pd.notna(r.get("earn_mult")) else np.nan,
            "earn_mult": r.get("earn_mult"), "pe_mult": r.get("pe_mult"), "pe0": r.get("pe0"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{W}/results/survival_table.csv", index=False)
    return df

def km_curve(df, label=""):
    """纯numpy实现KM估计，输出关键时点生存率。"""
    d = df.sort_values("duration_m")
    times = d["duration_m"].to_numpy()
    events = d["event"].to_numpy()
    uniq = np.unique(times[events == 1])
    n_at = len(d); surv = 1.0; out = []
    for t in uniq:
        d_t = int(((times == t) & (events == 1)).sum())
        n_t = int((times >= t).sum())
        if n_t > 0:
            surv *= (1 - d_t / n_t)
        out.append((t, surv))
    ks = {}
    for m in [6, 12, 24, 36, 48, 60]:
        s = [s for t, s in out if t <= m]
        ks[m] = s[-1] if s else 1.0
    print(f"[KM {label}] n={len(df)} events={int(events.sum())} "
          + " ".join(f"{m}月:{ks[m]*100:.0f}%" for m in ks), flush=True)
    return out

def main():
    df = build_survival_table()
    print(f"生存表: {len(df)} 家, 事件率 {df['event'].mean()*100:.1f}%")
    km_curve(df, "全体")
    for g, lab in [(df[df["fast"] == 1], "快十倍(<3年触及)"), (df[df["fast"] == 0], "慢十倍(≥3年触及)")]:
        if len(g) > 5: km_curve(g, lab)
    for g, lab in [(df[df["earn_dom"] == 1], "业绩主导"), (df[df["earn_dom"] == 0], "估值主导")]:
        if len(g) > 5: km_curve(g, lab)
    # Cox
    try:
        from lifelines import CoxPHFitter
        cox_df = df.dropna(subset=["years_to_touch"]).copy()
        cox_df["flood"] = ((cox_df["touch_year"] >= 2020) & (cox_df["touch_year"] <= 2021)).astype(int)
        cph = CoxPHFitter()
        cols = ["duration_m", "event", "years_to_touch", "flood"]
        sub = cox_df[cols + ["earn_dom"]].dropna()
        cph.fit(sub, duration_col="duration_m", event_col="event")
        cph.print_summary()
        cph.summary.to_csv(f"{W}/results/cox_summary.csv")
    except Exception as e:
        print("Cox skipped:", e)

if __name__ == "__main__":
    main()
