"""数据抓取器（持久版）：腾讯前复权日线 + 搜狐原始价量换手。断点续传。"""
import requests, time, os, json, threading, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_local = threading.local()

def _sess():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HDR)
    return _local.s

def tx_code(code6):
    if code6[0] in "69": return "sh" + code6
    if code6[0] in "48": return "bj" + code6
    return "sz" + code6

TX_BASE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

def fetch_tx_one(code6, start="2005-01-01", end="2026-07-18", fq="qfq",
                 max_pages=40, page_size=640, retry=4):
    sym = tx_code(code6)
    frames, cur_end = [], end
    for _ in range(max_pages):
        param = f"{sym},day,{start},{cur_end},{page_size},{fq}"
        j = None
        for attempt in range(retry):
            try:
                j = _sess().get(TX_BASE, params={"param": param}, timeout=15).json()
                break
            except Exception:
                if attempt == retry - 1: return None
                time.sleep(0.5 * (attempt + 1))
        data = j.get("data")
        if not isinstance(data, dict) or sym not in data or not data[sym]:
            return None if not frames else _mk_tx(frames)
        d = data[sym]
        key = f"{fq}day" if f"{fq}day" in d else ("day" if "day" in d else None)
        if key is None or not d[key]:
            break
        bars = d[key]
        df = pd.DataFrame([b[:6] for b in bars],
                          columns=["date", "open", "close", "high", "low", "volume"])
        frames.append(df)
        earliest = df["date"].iloc[0]
        if len(bars) < page_size or earliest <= start: break
        cur_end = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.03)
    return _mk_tx(frames) if frames else None

def _mk_tx(frames):
    df = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_sohu_one(code6, start="20050101", end="20260718", retry=4):
    u = (f"https://q.stock.sohu.com/hisHq?code=cn_{code6}"
         f"&start={start}&end={end}&stat=1&order=D&period=d")
    for attempt in range(retry):
        try:
            j = _sess().get(u, timeout=25).json()
            if not j or j[0].get("status") != 0: return None
            hq = [row[:10] for row in j[0]["hq"]]
            if not hq: return None
            df = pd.DataFrame(hq, columns=["date","open","close","chg","pct","low","high",
                                           "volume_shou","amount_wan","turnover_pct"])
            df = df.sort_values("date").reset_index(drop=True)
            for c in ["open","close","low","high","volume_shou","amount_wan"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["turnover_pct"] = pd.to_numeric(
                df["turnover_pct"].astype(str).str.rstrip("%"), errors="coerce")
            vol = df["volume_shou"] * 100.0
            tr = df["turnover_pct"] / 100.0
            df["float_shares"] = (vol / tr).where(tr > 0)
            df["float_mcap"] = df["close"] * df["float_shares"]
            return df[["date","open","close","high","low","volume_shou","amount_wan",
                       "turnover_pct","float_shares","float_mcap"]]
        except Exception:
            if attempt == retry - 1: return None
            time.sleep(0.6 * (attempt + 1))
    return None

def fetch_batch(codes, outdir, fn, workers=16, log_every=300, tag=""):
    os.makedirs(outdir, exist_ok=True)
    done = {f[:-4] for f in os.listdir(outdir) if f.endswith(".pkl")}
    todo = [c for c in codes if c not in done]
    print(f"[{tag}] total={len(codes)} cached={len(done)} todo={len(todo)}", flush=True)
    if not todo: return
    ok = fail = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                df = f.result()
                if df is not None and len(df):
                    df.to_pickle(f"{outdir}/{c}.pkl"); ok += 1
                else:
                    open(f"{outdir}/{c}.fail", "w").write("x"); fail += 1
            except Exception:
                open(f"{outdir}/{c}.fail", "w").write("x"); fail += 1
            if i % log_every == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ok={ok} fail={fail} {el:.0f}s "
                      f"eta={el/max(i,1)*(len(todo)-i):.0f}s", flush=True)
    print(f"[{tag}] done ok={ok} fail={fail}", flush=True)

if __name__ == "__main__":
    what, codes_file, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    fq = sys.argv[5] if len(sys.argv) > 5 else "qfq"
    codes = json.load(open(codes_file))
    if what == "tx":
        fn = lambda c: fetch_tx_one(c, fq=fq)
    else:
        fn = fetch_sohu_one
    fetch_batch(codes, outdir, fn, workers=workers, tag=what+":"+fq)
