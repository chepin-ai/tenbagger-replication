# Tenbagger Replication (420 十倍股研究复现)
Independent replication of "Three Markets, Twenty Years: Anatomy of 420 Ten-Baggers" with real market data.
- fetchers.py / fetch_hk.py: data pipelines (Tencent hfq daily incl. delisted, Sohu turnover->float mcap)
- detect2.py: ten-bagger detection engine (O(n) sliding-window, STRICT_MCAP, incremental save)
- survival.py: Kaplan-Meier + Cox PH ("permanent loss" event definition)
- identify.py: ex-ante identifiability logistic model
- eps_decomp.py: earnings x valuation decomposition (iFinD EPS)
- results/: all output tables; REPORT.md: full verification report (Chinese)
