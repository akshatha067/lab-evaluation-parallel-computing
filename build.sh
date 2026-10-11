#!/bin/bash
set -e
FLAGS="-O3 -march=native -ffast-math"
gcc   $FLAGS src/seq.c      -o seq
gcc   $FLAGS src/pthreads.c -o pthreads -pthread
mpicc $FLAGS src/mpi.c      -o mpi
echo "Build OK"
