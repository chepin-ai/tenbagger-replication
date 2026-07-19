"""港股全宇宙抓取：枚举 hk00001-hk09999（腾讯接口，含退市股）。"""
import requests, time, os, json, threading, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HDR = {"User-Agent": "Mozilla/5.0"}
_local = threading.local()
BASE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

def _sess():
    if not hasattr(_local, "s"):
        _local.s = requests.Session(); _local.s.headers.update(HDR)
    return _local.s

def fetch_hk_one(sym, start="2005-01-01", end="2026-07-18", fq="hfq",
                 max_pages=30, page_size=640):
    frames, cur_end = [], end
    for _ in range(max_pages):
        param = f"{sym},day,{start},{cur_end},{page_size},{fq}"
        j = None
        for attempt in range(3):
            try:
                j = _sess().get(BASE, params={"param": param}, timeout=12).json()
                break
            except Exception:
                if attempt == 2: return None
                time.sleep(0.4)
        data = j.get("data")
        if not isinstance(data, dict) or sym not in data or not data[sym]:
            return None if not frames else _mk(frames)
        d = data[sym]
        key = f"{fq}day" if f"{fq}day" in d else ("day" if "day" in d else None)
        if not key or not d[key]: break
        bars = d[key]
        df = pd.DataFrame([b[:6] for b in bars],
                          columns=["date","open","close","high","low","volume"])
        frames.append(df)
        if len(bars) < page_size or df["date"].iloc[0] <= start: break
        cur_end = (pd.Timestamp(df["date"].iloc[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.02)
    return _mk(frames) if frames else None

def _mk(frames):
    df = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    outdir = sys.argv[1]
    lo, hi = int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    os.makedirs(outdir, exist_ok=True)
    syms = [f"hk{i:05d}" for i in range(lo, hi + 1)]
    done = {f[:-4] for f in os.listdir(outdir) if f.endswith(".pkl")}
    null = {f[:-5] for f in os.listdir(outdir) if f.endswith(".null")}
    todo = [s for s in syms if s not in done and s not in null]
    print(f"range={lo}-{hi} todo={len(todo)}", flush=True)
    ok = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_hk_one, s): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                df = f.result()
                if df is not None and len(df):
                    df.to_pickle(f"{outdir}/{s}.pkl"); ok += 1
                else:
                    open(f"{outdir}/{s}.null", "w").write("x")
            except Exception:
                open(f"{outdir}/{s}.null", "w").write("x")
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} valid={ok} {el:.0f}s eta={el/i*(len(todo)-i):.0f}s", flush=True)
    print(f"done valid={ok}", flush=True)

if __name__ == "__main__":
    main()
