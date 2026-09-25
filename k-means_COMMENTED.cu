#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define BLKDIM 256

int n_dims;             /* number of dimensions.                */

int n_points;           /* number of data points.               */

int n_clusters;         /* number of clusters.                  */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid, which is an
                           array of `n_dims` floating point
                           numbers.                             */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points that belong to
                           cluster j.                           */

int *cluster_of;        /* [array of length n_points] `clusters_of[i]`
                           is the ID of the cluster assigned to the
                           i-th data point; cluster IDs are integer in
                           0..(n_clusters-1).                   */

/*Device pointers*/
float *dev_data;
float *dev_centroids;
float *dev_new_centroids;
int *dev_cluster_of;
int *dev_counts;

/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
// void vzero( float *p)
// {
//     for (int d=0; d<n_dims; d++)
//         p[d] = 0.0f;
// }

// /* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
//    have size `n_dims`. */
// void vadd( float *p1, const float *p2 )
// {
//     for (int d=0; d<n_dims; d++)
//         p1[d] += p2[d];
// }

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float *p, float v)   
{
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
__host__ __device__ float sqdist( float *p1, float *p2, int n_dims)
{
    float result = 0.0;
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
__host__ __device__ int IDX(int i, int d, int n_dims)
{
    return i*n_dims + d;
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy( &centroids[IDX(select,0, n_dims)], &data[IDX(i,0, n_dims)] );
        }
        remaining--;
    }
}

/* Assign each data point to the nearest centroid. Updates
   the `counts` array. */
__global__ void classify_kernel( float *dev_data,
                                float *dev_centroids,
                                int *dev_cluster_of,
                                int *dev_counts,
                                int n_points, int n_clusters, int n_dims
                             )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    

    extern __shared__ int blk_counts[];
    for (int idx = threadIdx.x; idx < n_clusters; idx += blockDim.x)
        blk_counts[idx] = 0;
    
    __syncthreads();

    if (i < n_points){
        int nearest = 0;
        float mindist = sqdist( &dev_data[IDX(i,0, n_dims)], &dev_centroids[IDX(nearest, 0, n_dims)], n_dims );
        for (int j=1; j<n_clusters; j++) {
                const float dist = sqdist( &dev_data[IDX(i, 0, n_dims)], &dev_centroids[IDX(j, 0, n_dims)], n_dims );
                if ( dist < mindist ) {
                    mindist = dist;
                    nearest = j;
                }
            }

        dev_cluster_of[i] = nearest;
        atomicAdd(&blk_counts[nearest], 1);
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < n_clusters; idx += blockDim.x)
        atomicAdd(&dev_counts[idx], blk_counts[idx]);
}

/* Update the centroids. Set the centroid of each cluster to the
   barycenter of the points. Returns the maximum shift, i.e., the
   maximum difference between the (squared) old and new position of
   all centroids. */
__global__ void accumulate_kernel( float *dev_data,
                                float *dev_new_centroids,
                                int *dev_cluster_of,
                                int n_points, int n_clusters, int n_dims )
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float blk_new_centroids[];
    for (int idx = threadIdx.x; idx < n_clusters*n_dims; idx += blockDim.x)
        blk_new_centroids[idx] = 0.0f;
    
    __syncthreads();
   
    if (i < n_points){
        int cluster_id = dev_cluster_of[i];
        for (int dim = 0; dim < n_dims; dim++)   
            atomicAdd(&blk_new_centroids[IDX(cluster_id, dim, n_dims)], dev_data[IDX(i, dim, n_dims)]);
    }

    __syncthreads();
   
    for (int idx = threadIdx.x; idx < n_clusters*n_dims; idx += blockDim.x)
            atomicAdd(&dev_new_centroids[idx], blk_new_centroids[idx]);

    
}

float average_new_centroids(void){
    float maxshift = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy( &new_centroids[IDX(j,0, n_dims)], &centroids[IDX(j,0, n_dims)] );
        } else {
            // We divide the accumulated sum in new_centroids by the number of points assigned to current cluster
            vmul( &new_centroids[IDX(j, 0, n_dims)], 1.0f/counts[j] );
        }
        // The max_shift refers to how much the CENTROIDS have overall moved between one iteration and the other
        const float shift = sqdist( &centroids[IDX(j, 0, n_dims)], &new_centroids[IDX(j, 0, n_dims)], n_dims );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0, n_dims)], &new_centroids[IDX(j, 0, n_dims)] );
    }
    
    return maxshift;
}


    
  


/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input( FILE *f )
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_points % n_dims == 0); /* If this assertion fails, then
                                       there is some line of the input
                                       file that has != n_dims
                                       items. */

    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d, n_dims)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids into a
   file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time. */
void save_centroids( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j=0; j<n_clusters; j++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d, n_dims)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters
   into a file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time.
*/
void save_clusters( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "out_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d, n_dims)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d, n_dims)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d, n_dims)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    const float TOL = 1e-5;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    //Initialize n_dims (D), n_points (N), *data with the datapoints in the input file
    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    // ******Parallelization Variables*******

    //Compute number of blocks according to the set number of threads per block
    const int num_blks = (n_points + BLKDIM - 1) / BLKDIM;
    
    // Compute size of shared SM memory
    size_t shared_memory_classify_kernel = n_clusters * sizeof(*counts);
    size_t shared_memory_accumulate_kernel = n_clusters * n_dims * sizeof(*centroids);

    // Allocate memory to hold HOST copies of variables 
    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    //Populate 'centroids' vectors with random points that will be used as the initial centroids for the first iteration
    init_centroids();

    //Allocate DEVICE Global Memory for device counterparts of the host variables.
    cudaSafeCall(cudaMalloc((void**)&dev_data, n_points * n_dims * sizeof(*dev_data)));
    cudaSafeCall(cudaMalloc((void**)&dev_centroids, n_clusters * n_dims * sizeof(*dev_centroids)));
    cudaSafeCall(cudaMalloc((void**)&dev_new_centroids, n_clusters * n_dims * sizeof(*dev_new_centroids)));
    cudaSafeCall(cudaMalloc((void**)&dev_cluster_of, n_points * sizeof(*dev_cluster_of)));
    cudaSafeCall(cudaMalloc((void**)&dev_counts, n_clusters  * sizeof(*dev_counts)));

    //Send dataset and initialized centroids to DEVICE memory
    cudaSafeCall(cudaMemcpy(dev_data, data, n_points  * n_dims  * sizeof(*data), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dev_centroids, centroids, n_clusters * n_dims * sizeof(*centroids), cudaMemcpyHostToDevice));

    
    //printf("Main loop starts\n\n");

    float shift;
    int iter = 0;

    // Set timers
    double t_classify = 0.0;
    double t_accumulate = 0.0;
    double t_serial_comm = 0.0;

    const double tstart = hpc_gettime();
    do {

        double t0;
        t0 = hpc_gettime();

        // Initialize DEVICE copy of 'counts' array to 0 before executing each iteration.
        cudaSafeCall(cudaMemset(dev_counts, 0, n_clusters * sizeof(*counts)));

        // Launch the 'classify' kernel on the DEVICE (Parallel Execution)
        // Each thread is in charge of one single datapoint 'i' among the n_points (N) total.
        // Threads are divided into blocks, each block contains BLKDIM (256) threads
        classify_kernel<<<num_blks, BLKDIM, shared_memory_classify_kernel>>>(
            dev_data, dev_centroids, dev_cluster_of, dev_counts,
            n_points, n_clusters, n_dims
        );
        // Wait for each thread to synchronize
        cudaDeviceSynchronize();
        cudaCheckError();
        t_classify += (hpc_gettime() - t0);

        /* The following lines are useful only if you want to generate
           a movie of the evolution of the algorithm; if you are
           taking times for performance evaluation purposes, remove
           these lines, otherwise the time will be dominated by I/O
           operations. */
//#ifdef MAKE_MOVIE
//          save_centroids(iter);
//          save_clusters(iter);
//#endif
        t0 = hpc_gettime();
        // Initialize DEVICE copy of 'new_centroids' array to 0 before executing each iteration.
        cudaSafeCall(cudaMemset(dev_new_centroids, 0, n_clusters * n_dims * sizeof(*new_centroids)));

        // Launch the 'accumulate' kernel on the DEVICE (Parallel Execution).
        // This kernel is in charge of executing only one part of the computation that was executed by the 'update_centroids' function of the serial program.
        // In particular, it accumulates the coordinates of each datapoint 'i' into the device copy of the 'new_centroids' vector, at the right index according to the assigned cluster.
        // The normalization of the accumulated sum by the cluster size happens serially, after this kernel has been executed
        accumulate_kernel<<<num_blks, BLKDIM, shared_memory_accumulate_kernel>>>(
            dev_data, dev_new_centroids, dev_cluster_of, 
            n_points, n_clusters, n_dims);
        cudaDeviceSynchronize();
        cudaCheckError();
        t_accumulate += (hpc_gettime() - t0);
        
        t0 = hpc_gettime();
        // To normalize the accumulated in 'dev_new_centroids', we must copy the content of such vector back to the host memory, together with 'counts', which holds the cluster sizes.
        cudaSafeCall(cudaMemcpy(new_centroids, dev_new_centroids, n_clusters * n_dims * sizeof(*new_centroids), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(counts, dev_counts, n_clusters * sizeof(*counts), cudaMemcpyDeviceToHost));
       
        //SERIAL function that normalizes the sums accumulated in new_centroids, assigns it to 'centroids', and calculates the current shift.
        shift = average_new_centroids();         
        t_serial_comm += (hpc_gettime() - t0);

        // Copy the new centroids onto device memory for next iteration
        cudaSafeCall(cudaMemcpy(dev_centroids, centroids, n_clusters * n_dims * sizeof(*centroids), cudaMemcpyHostToDevice));
        
        //printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;

    } while ( /*(shift > TOL) &&*/ (iter <= MAXITER) );     //STOPPING CONDITION
    const double elapsed = hpc_gettime() - tstart;

    //Copy the centroids and cluster assignments of the last iteration, from device memory to host memory. These are the final results of the algorithms and will be used by 'save_results' function 
    cudaSafeCall(cudaMemcpy(centroids, dev_centroids, n_clusters * n_dims * sizeof(*centroids), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(cluster_of,  dev_cluster_of, n_points * sizeof(*cluster_of),   cudaMemcpyDeviceToHost));


    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n\n", elapsed);

    save_results(outputf);

    fclose(outputf);

    // Print results&stats
    printf("--- Execution Statistics ---\n");
    printf("Input file       : %s\n", argv[2]);
    printf("Data points (N)  : %d\n", n_points);
    printf("Dimensions (D)   : %d\n", n_dims);
    printf("Clusters (K)     : %d\n", n_clusters);
    printf("Output file      : %s\n", argv[3]);
    printf("--------------------------------");
    printf("Total loop time  : %f s\n", elapsed);
    printf("Classify time    : %f s\n", t_classify);
    printf("Accumulate time  : %f s\n", t_accumulate);
    printf("Serial/Comm time : %f s\n", t_serial_comm);
    printf("----------------------------\n\n");

    free(data);
    free(centroids);
    free(cluster_of);
    free(counts);

    cudaSafeCall(cudaFree(dev_data));
    cudaSafeCall(cudaFree(dev_centroids));
    cudaSafeCall(cudaFree(dev_new_centroids));
    cudaSafeCall(cudaFree(dev_cluster_of));
    cudaSafeCall(cudaFree(dev_counts));
    

    cudaSafeCall(cudaMalloc((void**)&dev_data, n_points * n_dims * sizeof(*dev_data)));
    cudaSafeCall(cudaMalloc((void**)&dev_centroids, n_clusters * n_dims * sizeof(*dev_centroids)));
    cudaSafeCall(cudaMalloc((void**)&dev_new_centroids, n_clusters * n_dims * sizeof(*dev_new_centroids)));
    cudaSafeCall(cudaMalloc((void**)&dev_cluster_of, n_points * sizeof(*dev_cluster_of)));
    cudaSafeCall(cudaMalloc((void**)&dev_counts, n_clusters  * sizeof(*dev_counts)));




    return EXIT_SUCCESS;
}
