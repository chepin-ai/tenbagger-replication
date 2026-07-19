#!/bin/bash
W=/mnt/agents/work/tenbagger
mkdir -p $W/bak_hfq $W/bak_mcap
for i in $(seq 1 200); do
  cp -rn $W/bak_hfq/. $W/data_hfq/ 2>/dev/null
  cp -rn $W/bak_mcap/. $W/data_mcap/ 2>/dev/null
  n1=$(ls $W/data_hfq/*.pkl 2>/dev/null | wc -l)
  n2=$(ls $W/data_mcap/*.pkl 2>/dev/null | wc -l)
  echo "[sup $i] hfq=$n1 mcap=$n2"
  if [ "$n1" -ge 5450 ] && [ "$n2" -ge 5450 ]; then echo ALL-DONE; break; fi
  if [ "$n1" -lt 5450 ]; then python3 $W/code/fetchers.py tx $W/codes_main.json $W/data_hfq 20 hfq; fi
  if [ "$n2" -lt 5450 ]; then python3 $W/code/fetchers.py sohu $W/codes_main.json $W/data_mcap 10; fi
  cp -rn $W/data_hfq/. $W/bak_hfq/ 2>/dev/null
  cp -rn $W/data_mcap/. $W/bak_mcap/ 2>/dev/null
  sleep 2
done
