#!/bin/bash
# Usage: ./run_experiments.sh 10 15      (sizes in GB)
set -e
[ $# -lt 1 ] && { echo "Usage: $0 <size_GB> [<size_GB> ...]"; exit 1; }
WORKERS=${WORKERS:-"1 2 4 8 16"}
mkdir -p results
OUT=results/results.csv
[ -f $OUT ] || echo "method,gb,workers,time,checksum" > $OUT
for GB in "$@"; do
  echo "===== $GB GB ====="
  ./seq $GB | tee -a $OUT
  for P in $WORKERS; do
    ./pthreads $GB $P | tee -a $OUT
    mpirun --oversubscribe -np $P ./mpi $GB | tee -a $OUT
  done
done
echo "Done. Results in $OUT"
