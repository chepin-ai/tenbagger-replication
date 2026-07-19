"""通过 iFinD 插件拉取美股/港股个股长历史前复权日线（3年一段拼接）。"""
import subprocess, sys, os, json, time
import pandas as pd

PLUGIN = "/app/.agents/plugins/ifind"

def ifind_price(ticker, start, end, adjust="forward", cache_dir="/mnt/agents/work/tenbagger/ifind_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    fp = f"{cache_dir}/{ticker.replace('.','_')}_{start}_{end}.csv"
    if not os.path.exists(fp):
        params = {"ticker": ticker, "start_date": start, "end_date": end,
                  "file_path": fp, "adjust": adjust}
        r = subprocess.run([sys.executable, f"{PLUGIN}/scripts/ifind_tool.py", "call",
                            "--api-name", "ifind_get_price",
                            "--params-json", json.dumps(params)],
                           capture_output=True, text=True, timeout=180)
        if not os.path.exists(fp):
            print("FAIL", ticker, start, r.stdout[-300:], r.stderr[-200:])
            return None
        time.sleep(0.5)
    return fp

def ifind_price_long(ticker, y0=2005, y1=2026, adjust="forward"):
    fps = []
    y = y0
    while y <= y1:
        s, e = f"{y}-01-01", f"{min(y+2, y1)}-12-31"
        fp = ifind_price(ticker, s, e, adjust)
        if fp: fps.append(fp)
        y += 3
    dfs = []
    for fp in fps:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception:
            pass
    if not dfs: return None
    df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    return df

if __name__ == "__main__":
    ticker = sys.argv[1]
    df = ifind_price_long(ticker)
    if df is not None:
        print(ticker, len(df), df.columns.tolist()[:8])
        print(df.head(2).to_string())
        out = f"/mnt/agents/work/tenbagger/ifind_{ticker.replace('.','_')}_full.csv"
        df.to_csv(out, index=False)
        print("saved:", out)
