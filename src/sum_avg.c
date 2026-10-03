#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double sequential_sum(const double *a, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i++) s += a[i];
    return s;
}

static double parallel_sum(const double *a, long n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) s += a[i];
    return s;
}

static void generate(double *a, long n) {
    unsigned long long x = 88172645463325252ULL;
    for (long i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        a[i] = (double)(x % 1000000ULL) / 1000.0;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <N> <threads> <repeats>\n", argv[0]);
        return 1;
    }
    long n   = atol(argv[1]);
    int  thr = atoi(argv[2]);
    int  rep = atoi(argv[3]);

    double *a = malloc(n * sizeof(double));
    if (!a) { fprintf(stderr, "malloc failed\n"); return 1; }
    generate(a, n);
    omp_set_num_threads(thr);

    volatile double w = parallel_sum(a, n); (void)w;

    double seq_s = 0, par_s = 0, seq_t = 0, par_t = 0;
    for (int r = 0; r < rep; r++) {
        double t0 = omp_get_wtime();
        seq_s = sequential_sum(a, n);
        seq_t += omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        par_s = parallel_sum(a, n);
        par_t += omp_get_wtime() - t0;
    }
    seq_t /= rep; par_t /= rep;

    double speedup = seq_t / par_t;
    double eff = speedup / thr;
    double err = fabs(seq_s - par_s) / fabs(seq_s);

    printf("%ld,%d,%.6f,%.6f,%.3f,%.3f,%.6f,%.6f,%.3e\n",
           n, thr, seq_t, par_t, speedup, eff, seq_s / n, par_s / n, err);
    free(a);
    return 0;
}
