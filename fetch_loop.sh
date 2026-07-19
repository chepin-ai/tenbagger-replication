#!/bin/bash
# 自重启抓取循环：沙箱重置后重跑本脚本即可续传
W=/mnt/agents/work/tenbagger
for i in $(seq 1 30); do
  n=$(ls $W/data_qfq/*.pkl 2>/dev/null | wc -l)
  echo "[loop $i] qfq done=$n"
  if [ "$n" -ge 6000 ]; then break; fi
  python3 $W/code/fetchers.py tx $W/codes.json $W/data_qfq 20
  sleep 3
done
for i in $(seq 1 30); do
  n=$(ls $W/data_mcap/*.pkl 2>/dev/null | wc -l)
  echo "[loop $i] mcap done=$n"
  if [ "$n" -ge 6000 ]; then break; fi
  python3 $W/code/fetchers.py sohu $W/codes.json $W/data_mcap 12
  sleep 3
done
echo ALL-DONE
