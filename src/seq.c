/* Sequential baseline. Usage: ./seq <dataset_size_in_GB> */
#include <time.h>
#include "common.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <dataset_size_in_GB>\n", argv[0]); return 1; }
    double gb = atof(argv[1]);
    long n = samples_for_gb(gb);

    float *buf = malloc((size_t)BLOCK_SAMPLES * D_FEAT * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    init_weights();

    double t0 = now();
    double checksum = process_range(buf, 0, n);
    double t = now() - t0;

    printf("seq,%.2f,1,%.4f,%.6e\n", gb, t, checksum);
    free(buf);
    return 0;
}
