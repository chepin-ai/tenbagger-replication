"""业绩×估值分解：对十倍股候选，取起点财年/触及财年EPS（iFinD），
配合原始价格（搜狐）计算 PE0→PE1 与 EPS0→EPS1。
守住组 vs 失守组中位数对照原文 5.9×2.2 / 2.6×4.3。增量落盘。"""
import subprocess, sys, os, json, time
import pandas as pd
import numpy as np

W = "/mnt/agents/work/tenbagger"
PLUGIN = "/app/.agents/plugins/ifind"
OUT = f"{W}/results/eps_decomp.csv"

def batch_eps(tickers, fy, cache):
    """tickers: list of XXXXXX.SH/SZ; fy: '20161231'. 返回 {ticker: eps}"""
    res = {}
    todo = [t for t in tickers if (t, fy) not in cache]
    for i in range(0, len(todo), 3):
        grp = todo[i:i+3]
        fp = f"/tmp/eps_{fy}_{i}.csv"
        params = {"ticker": ",".join(grp), "financial_parameter": fy,
                  "category": "profitability", "file_path": fp}
        try:
            subprocess.run([sys.executable, f"{PLUGIN}/scripts/ifind_tool.py", "call",
                            "--api-name", "ifind_get_stock_financial_index",
                            "--params-json", json.dumps(params)],
                           capture_output=True, text=True, timeout=120)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                codecol = [c for c in df.columns if "thscode" in c.lower()]
                for _, row in df.iterrows():
                    t = row[codecol[0]] if codecol else grp[0]
                    cache[(t, fy)] = row.get("ths_eps_basic_stock", np.nan)
                os.remove(fp)
        except Exception:
            pass
        time.sleep(0.4)
    return {t: cache.get((t, fy), np.nan) for t in tickers}

def ifind_code(c):
    return f"{c}.SH" if c[0] == "6" else f"{c}.SZ"

def main():
    fin = pd.read_csv(f"{W}/results/tenb_10x_strict.csv")
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin = fin.dropna(subset=["touch_date"])
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["code"].astype(str).str.zfill(6))
    cache = {}
    rows = []
    for k, (_, r) in enumerate(fin.iterrows(), 1):
        c = r["code"]
        if c in done: continue
        fy0 = str(pd.Timestamp(r["first_q"]).year) + "1231"
        fy1 = str(pd.Timestamp(r["touch_date"]).year) + "1231"
        t = ifind_code(c)
        e0 = batch_eps([t], fy0, cache).get(t, np.nan)
        e1 = batch_eps([t], fy1, cache).get(t, np.nan)
        # 原始价格
        p0 = p1 = np.nan
        mcp = f"{W}/data_mcap/{c}.pkl"
        if os.path.exists(mcp):
            mc = pd.read_pickle(mcp).set_index("date")
            try: p0 = mc.loc[r["first_q"], "close"]
            except Exception: pass
            try: p1 = mc.loc[r["touch_date"], "close"]
            except Exception: pass
        row = dict(code=c, fy0=fy0, fy1=fy1, eps0=e0, eps1=e1, p0=p0, p1=p1,
                   hold=r["hold"], years_to_touch=r["years_to_touch"])
        try:
            if all(np.isfinite([e0, e1, p0, p1])) and e0 > 0 and e1 > 0:
                pe0, pe1 = p0 / e0, p1 / e1
                row.update(earn_mult=e1 / e0, pe_mult=pe1 / pe0,
                           product=(e1 / e0) * (pe1 / pe0), pe0=pe0, pe1=pe1)
        except Exception:
            pass
        rows.append(row)
        if k % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"{k}/{len(fin)}", flush=True)
    if rows:
        old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("code").to_csv(OUT, index=False)
    print("done", len(rows))

if __name__ == "__main__":
    main()
