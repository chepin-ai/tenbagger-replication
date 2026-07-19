import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

W = "/mnt/agents/work/tenbagger"
df = pd.read_csv(f"{W}/results/identify_dataset.csv").dropna()
d = df["date"].str
df["qtr"] = d[:4] + "Q" + ((d[5:7].astype(int)-1)//3+1).astype(str)
y = df["y"].to_numpy()
n = len(df)
print(f"n={n}, pos={y.sum()}, quarters={df['qtr'].nunique()}")

def llf(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float((y*np.log(p) + (1-y)*np.log(1-p)).sum())

p0 = y.mean()
ll0 = llf(np.full(n, p0))
# M1 特征（statsmodels可收敛）
X1 = sm.add_constant(df[["log_mcap","log_amt","ret252","vol252","dd52"]])
m1 = sm.Logit(df["y"], X1).fit(disp=0)
ll1 = m1.llf
# M2 季度FE：纯哑变量模型→组内经验率即MLE
pq = df.groupby("qtr")["y"].transform("mean").to_numpy()
ll2 = llf(pq)
# M3 特征+季度哑变量（L2, C大≈弱正则）
Xq = pd.get_dummies(df["qtr"], drop_first=True).astype(float)
X3 = np.column_stack([X1.to_numpy(), Xq.to_numpy()])
lr = LogisticRegression(C=1e4, max_iter=2000).fit(X3, y)
ll3 = llf(lr.predict_proba(X3)[:,1])
df3 = X3.shape[1]

def r2(ll): return 1-ll/ll0
print(f"伪R2: 特征={r2(ll1):.4f} 季度FE={r2(ll2):.4f} 全模型={r2(ll3):.4f}")
lr_when = 2*(ll3-ll1); df_when = Xq.shape[1]
lr_what = 2*(ll3-ll2); df_what = 5
print(f"LR 加季度(When): chi2={lr_when:.0f}(df={df_when}) p={1-chi2.cdf(lr_when,df_when):.2e}")
print(f"LR 加特征(What): chi2={lr_what:.0f}(df={df_what}) p={1-chi2.cdf(lr_what,df_what):.4f}")
print(f"变异解释比 When:What = {r2(ll2)/r2(ll1):.1f} : 1")
fe_span = np.log(pq.clip(1e-9)/(1-pq.clip(1e-9)))
fe_span = fe_span.max()-fe_span.min()
b = np.abs(m1.params.drop("const"))
span = (b*df[["log_mcap","log_amt","ret252","vol252","dd52"]].std()).sum()
print(f"季度效应极差={fe_span:.2f} vs 特征±1σ合计={span:.2f} → When/What幅度比={fe_span/span:.1f}x")
open(f"{W}/results/var_decomp.txt","w").write(
 f"R2 feat={r2(ll1):.4f} qtrFE={r2(ll2):.4f} full={r2(ll3):.4f}\n"
 f"When:What R2 ratio={r2(ll2)/r2(ll1):.1f}; LR when p={1-chi2.cdf(lr_when,df_when):.2e}; LR what p={1-chi2.cdf(lr_what,df_what):.4f}\n"
 f"FE range={fe_span:.2f}; feat span={span:.2f}; amplitude ratio={fe_span/span:.1f}\n")
