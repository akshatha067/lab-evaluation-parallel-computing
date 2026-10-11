# Sequential vs Pthreads vs MPI on a Large AI Workload (10 GB)

**Team 3 | Parallel Computing Mini-Project (Lab Evaluation)**
**Team members:** Akshatha 150, Kavya 169, Anusha 166

## Why this version
The earlier sum/average test on 0.5-4 GB was memory-bound: all versions gave similar times (speedup stuck near 2.5x).
This version uses a compute-bound AI workload on a 10 GB dataset, comparing Sequential, Pthreads and MPI.

## Workload
Batch neural-network inference. Each sample is 256 floats (1 KB). For every sample:
a 128-neuron hidden layer with ReLU is evaluated, then an output dot product is taken.
10 GB is about 10.5 million samples and about 3.4e11 multiply-adds.
Data is created in code with a fixed hash (identical for all versions) and streamed in 16 MB blocks per worker,
so a 10 GB dataset runs on a laptop with 7.6 GB RAM. The checksum (sum of all outputs) is identical across all versions.

## Folder structure
- src/seq.c : sequential baseline
- src/pthreads.c : threads (pthread_create / pthread_join, partial sums added by main)
- src/mpi.c : processes (each handles a slice, MPI_Reduce adds partial sums)
- src/common.h : shared data generator and compute kernel
- build.sh : builds all three with the same flags
- run_experiments.sh : runs all experiments, writes results/results.csv
- plot.py : creates graphs/ and results/summary.csv
- results/ : raw timings and summary (speedup, efficiency)
- graphs/ : time, speedup and efficiency graphs
- presentation/ : lab evaluation PPT
- old_sum_avg_openmp/ : earlier sum/average experiment (OpenMP)

## Build and run
    ./build.sh
    ./seq 10                      # 10 GB sequential
    ./pthreads 10 8               # 10 GB, 8 threads
    mpirun -np 8 ./mpi 10         # 10 GB, 8 processes
    ./run_experiments.sh 10       # full experiment -> results/results.csv
    python3 plot.py               # graphs and summary

## Setup
Intel Core Ultra 5 125H, 7.6 GB RAM, WSL Ubuntu, GCC with -O3 -march=native -ffast-math. One run per configuration.
Timed region: creating each block plus running the network. Speedup = Tseq / Tpar. Efficiency = speedup / workers.

## Results (10 GB, sequential = 49.67 s)

| Method | Workers | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| Pthreads | 1 | 48.624 | 1.02 | 1.02 |
| MPI | 1 | 50.027 | 0.99 | 0.99 |
| Pthreads | 2 | 24.667 | 2.01 | 1.01 |
| MPI | 2 | 26.201 | 1.90 | 0.95 |
| Pthreads | 4 | 13.894 | 3.58 | 0.89 |
| MPI | 4 | 14.421 | 3.45 | 0.86 |
| Pthreads | 8 | 8.248 | 6.02 | 0.75 |
| MPI | 8 | 8.460 | 5.87 | 0.73 |
| Pthreads | 16 | 6.413 | 7.75 | 0.48 |
| MPI | 16 | 7.373 | 6.74 | 0.42 |

Graphs: graphs/time_10GB.png, graphs/speedup.png, graphs/efficiency.png

## Analysis
- Speedup is close to ideal up to 4 workers (efficiency 0.86-0.89) because the workload is compute-bound.
- At 8 and 16 workers efficiency falls to about 0.75 and 0.45. Likely causes (not measured): the CPU mixes performance and efficiency cores, and workers share cache, memory and overhead.
- Pthreads and MPI perform similarly; MPI is slightly slower, probably from process start-up and the reduce step.
- Compared with the earlier sum test (about 2.5x), this workload shows a clear, visible speedup: 49.7 s down to 6.4 s (7.7x).

## Conclusion
A compute-heavy AI workload on a large dataset shows the benefit of parallelism that a memory-bound sum hides.
Both Pthreads and MPI scale well, with diminishing returns beyond 4 workers on this laptop CPU.
