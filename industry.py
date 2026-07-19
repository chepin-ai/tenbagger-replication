"""开放问题c：十倍股的行业/产业链聚集（iFinD主营业务 → 关键词聚类）。"""
import subprocess, sys, os, json, time
import pandas as pd

W = "/mnt/agents/work/tenbagger"
PLUGIN = "/app/.agents/plugins/ifind"
OUT = f"{W}/results/industry.csv"

CLUSTERS = {
    "半导体/电子": ["半导体","集成电路","芯片","电子","封装","晶圆","显示","LED","PCB","印制"],
    "新能源/电池": ["锂","电池","新能源","光伏","太阳能","风电","充电","储能","钴","正极","负极","隔膜","电解"],
    "白酒/食品饮料": ["白酒","酒","食品","饮料","乳","调味","啤酒","肉","茶"],
    "医药生物": ["药","医","生物","疫苗","诊断","器械","医院","血制品","中药"],
    "互联网/软件": ["互联网","软件","信息","网络","游戏","数字","云","数据","IT","广告"],
    "化工/新材料": ["化工","材料","纤","塑","涂料","玻","钛","化学","硅胶","石墨"],
    "机械/装备制造": ["机械","装备","重工","机床","机器人","自动化","工程","泵","阀","电梯","叉车"],
    "汽车/零部件": ["汽车","车","零部件","轮胎","整车"],
    "有色金属/采矿": ["铜","铝","锌","镍","锡","钼","钨","稀土","矿","有色","黄金","煤","钢铁"],
    "房地产/建筑": ["房地产","地产","建筑","建材","水泥","装饰","园林"],
    "金融": ["银行","证券","保险","信托","金融","期货"],
    "军工": ["航空","航天","军","船舶","雷达","兵器"],
    "通信": ["通信","光模块","5G","光纤","基站"],
    "消费/家电零售": ["家电","零售","商贸","服装","纺","家居","化妆品","免税","超市"],
    "电力/公用": ["电力","水电","核电","公用","燃气","水务","环保","公交"],
    "农业": ["农业","养殖","种植","饲料","种子","猪","鸡"],
}

def classify(text):
    if not isinstance(text, str): return "其他"
    hits = {}
    for cl, kws in CLUSTERS.items():
        hits[cl] = sum(1 for k in kws if k in text)
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "其他"

def main():
    fin = pd.read_csv(f"{W}/results/tenb_10x_strict.csv")
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    done = {}
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        done = dict(zip(old["code"].astype(str).str.zfill(6), old["cluster"]))
    rows = []
    todo = [c for c in fin["code"] if c not in done]
    for i in range(0, len(todo), 3):
        grp = todo[i:i+3]
        tickers = [f"{c}.SH" if c[0]=="6" else f"{c}.SZ" for c in grp]
        fp = f"/tmp/ind_{i}.csv"
        try:
            subprocess.run([sys.executable, f"{PLUGIN}/scripts/ifind_tool.py", "call",
                            "--api-name", "ifind_get_stock_info",
                            "--params-json", json.dumps({"ticker": ",".join(tickers), "file_path": fp})],
                           capture_output=True, text=True, timeout=120)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                for _, r in df.iterrows():
                    code = str(r.get("thscode","")).split(".")[0]
                    biz = str(r.get("ths_main_businuess_stock","")) + str(r.get("ths_mo_product_name_stock",""))
                    rows.append({"code": code, "biz": biz[:80], "cluster": classify(biz)})
                os.remove(fp)
        except Exception: pass
        time.sleep(0.4)
        print(f"{i+len(grp)}/{len(todo)}", flush=True)
    new = pd.DataFrame(rows)
    old = pd.DataFrame([{"code":k,"cluster":v} for k,v in done.items()])
    alldf = pd.concat([old, new]).drop_duplicates("code")
    alldf.to_csv(OUT, index=False)
    print("\n行业聚集分布:")
    print(alldf["cluster"].value_counts().to_string())

if __name__ == "__main__":
    main()
