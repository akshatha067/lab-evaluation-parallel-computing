/* Pthreads version. Usage: ./pthreads <dataset_size_in_GB> <threads> */
#include <pthread.h>
#include <time.h>
#include "common.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }

typedef struct { long first, count; double result; } Task;

static void *worker(void *arg)
{
    Task *t = (Task *)arg;
    float *buf = malloc((size_t)BLOCK_SAMPLES * D_FEAT * sizeof(float));
    t->result = process_range(buf, t->first, t->count);
    free(buf);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "Usage: %s <dataset_size_in_GB> <threads>\n", argv[0]); return 1; }
    double gb = atof(argv[1]);
    int nt = atoi(argv[2]);
    long n = samples_for_gb(gb);
    init_weights();

    pthread_t *th = malloc(nt * sizeof(pthread_t));
    Task *tasks = malloc(nt * sizeof(Task));
    long chunk = n / nt;
    for (int i = 0; i < nt; i++) {
        tasks[i].first = i * chunk;
        tasks[i].count = (i == nt - 1) ? n - i * chunk : chunk;
    }

    double t0 = now();
    for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, worker, &tasks[i]);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    double checksum = 0.0;
    for (int i = 0; i < nt; i++) checksum += tasks[i].result;
    double t = now() - t0;

    printf("pthreads,%.2f,%d,%.4f,%.6e\n", gb, nt, t, checksum);
    free(th); free(tasks);
    return 0;
}
