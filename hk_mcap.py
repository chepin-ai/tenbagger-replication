"""港股候选市值过滤：起点日原始收盘 × iFinD总股本快照 ≈ 起点市值，≥10亿美元保留。"""
import os, sys, json, time, subprocess
import pandas as pd
import importlib.util

W = "/mnt/agents/work/tenbagger"
HKDUSD = 7.78
spec = importlib.util.spec_from_file_location("fhk", f"{W}/code/fetch_hk.py")
fhk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fhk)
PLUGIN = "/app/.agents/plugins/ifind"

def ifind_shares(syms4):
    """syms4: ['0700.HK',...] → {sym: total_shares}"""
    out = {}
    for i in range(0, len(syms4), 3):
        grp = syms4[i:i+3]
        fp = f"/tmp/hk_info_{i}.csv"
        params = {"ticker": ",".join(grp), "file_path": fp}
        try:
            subprocess.run([sys.executable, f"{PLUGIN}/scripts/ifind_tool.py", "call",
                            "--api-name", "ifind_get_stock_info",
                            "--params-json", json.dumps(params)],
                           capture_output=True, text=True, timeout=120)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                for _, row in df.iterrows():
                    t = row.get("thscode")
                    sh = row.get("ths_total_shares_stock")
                    if t: out[t] = sh
                os.remove(fp)
        except Exception: pass
        time.sleep(0.4)
    return out

def raw_close_at(sym, date_str):
    df = fhk.fetch_hk_one(sym, start="2005-01-01", end="2026-07-18", fq="")
    if df is None: return None
    df = df.set_index("date")
    try: return float(df.loc[date_str, "close"])
    except Exception:
        # 找最近交易日
        idx = df.index[df.index <= date_str]
        return float(df.loc[idx[-1], "close"]) if len(idx) else None

def main():
    cand = pd.read_csv(f"{W}/results/tenb_hk.csv")
    shares_file = f"{W}/results/hk_shares.json"
    shares = json.load(open(shares_file)) if os.path.exists(shares_file) else {}
    todo = [s for s in cand["sym"] if f"{s[2:]}.HK" not in shares]
    if todo:
        new = ifind_shares([f"{s[2:]}.HK" for s in todo])
        shares.update(new)
        json.dump(shares, open(shares_file, "w"))
    rows = []
    for _, r in cand.iterrows():
        sym = r["sym"]
        sh = shares.get(f"{sym[2:]}.HK")
        p0 = raw_close_at(sym, r["first_q"])
        mcap_usd = sh * p0 / HKDUSD if (sh and p0 and sh > 0) else None
        rows.append({**r.to_dict(), "shares": sh, "raw_p0": p0, "mcap_usd": mcap_usd})
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    df.to_csv(f"{W}/results/tenb_hk_mcap.csv", index=False)
    big = df[df["mcap_usd"] >= 1e9]
    print(f"候选 {len(df)} → 市值≥10亿美元 {len(big)} 家 (原文40家)")
    print(big.sort_values("n_qual", ascending=False).head(25)[
        ["sym","n_qual","first_q","last_q","years_to_touch","hold","mcap_usd"]].to_string(index=False))

if __name__ == "__main__":
    main()
