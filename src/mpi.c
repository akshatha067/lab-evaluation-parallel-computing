/* MPI version. Usage: mpirun -np <P> ./mpi <dataset_size_in_GB> */
#include <mpi.h>
#include "common.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) { if (!rank) fprintf(stderr, "Usage: mpirun -np P %s <dataset_size_in_GB>\n", argv[0]); MPI_Finalize(); return 1; }
    double gb = atof(argv[1]);
    long n = samples_for_gb(gb);

    long chunk = n / size;
    long first = rank * chunk;
    long count = (rank == size - 1) ? n - first : chunk;

    float *buf = malloc((size_t)BLOCK_SAMPLES * D_FEAT * sizeof(float));
    if (!buf) { fprintf(stderr, "rank %d: malloc failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    init_weights();

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double local = process_range(buf, first, count);
    double checksum = 0.0;
    MPI_Reduce(&local, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t = MPI_Wtime() - t0;
    double tmax;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("mpi,%.2f,%d,%.4f,%.6e\n", gb, size, tmax, checksum);
    free(buf);
    MPI_Finalize();
    return 0;
}
