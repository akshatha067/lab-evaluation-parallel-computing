#!/bin/bash
set -e
mkdir -p results
gcc -O2 -fopenmp src/sum_avg.c -o sum_avg -lm
OUT=results/results.csv
echo "n,threads,seq_time,par_time,speedup,efficiency,seq_avg,par_avg,rel_error" > $OUT
for N in 1000000 10000000 50000000 100000000; do
  for T in 1 2 4 8; do
    ./sum_avg $N $T 5 | tee -a $OUT
  done
done
echo "Done. Results in $OUT"
