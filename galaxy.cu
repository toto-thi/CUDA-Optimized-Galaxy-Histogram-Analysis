#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

float *ra_real, *decl_real; // data for the real galaxies will be read into these arrays
int NoofReal;               // number of real galaxies

float *ra_sim, *decl_sim; // data for the simulated random galaxies will be read into these arrays
int NoofSim;              // number of simulated random galaxies

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

const int numBins = totaldegrees * binsperdegree;
const float rad = 180.0f / M_PI;

__global__ void calculateHistograms(float *ra1, float *dec1, int NoofReal, float *ra2, float *dec2, int NoofSim, unsigned int *histogram)
{
    __shared__ unsigned int sharedHistogram[numBins];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    while (localIdx < numBins)
    {
        sharedHistogram[localIdx] = 0U;
        localIdx += threadsperblock;
    }

    __syncthreads();

    // calculate Histogram using shared memory
    if (idx < NoofReal)
    {
        float cosDec1 = cosf(dec1[idx]);
        float sinDec1 = sinf(dec1[idx]);
        float ra1Val = ra1[idx];
        float degree = 0.0;

        for (int i = 0; i < NoofReal; i += threadsperblock)
        {
            for (int j = 0; j < min(threadsperblock, NoofReal - i); j++)
            {
                float angle = cosDec1 * cosf(dec2[i + j]) * cosf(ra1Val - ra2[i + j]) + sinDec1 * sinf(dec2[i + j]);
                angle = fminf(1.0f, fmaxf(-1.0f, angle));
                degree = acos(angle) * rad;
                int bins = (int)(degree * 4.0f);
                atomicAdd(&sharedHistogram[bins], 1);
            }
        }
    }

    __syncthreads();

    localIdx = threadIdx.x;
    // Atomic add to global memory
    while (localIdx < numBins)
    {
        atomicAdd(&histogram[localIdx], sharedHistogram[localIdx]);
        localIdx += threadsperblock;
    }
}

int main(int argc, char *argv[])
{
    int noofblocks;
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);
    unsigned long int histogramDRsum = 0L, histogramDDsum = 0L, histogramRRsum = 0L; // initialized histogram to 0
    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;
    cudaError_t myError;

    FILE *outfil;

    if (argc != 4)
    {
        printf("Usage: a.out real_data random_data output_data\n");
        return (-1);
    }

    if (getDevice(0) != 0)
        return (-1);

    if (readdata(argv[1], argv[2]) != 0)
        return (-1);

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // allocate mameory on the GPU
    float *d_ra_real, *d_dec_real;
    float *d_ra_sim, *d_dec_sim;
    cudaMalloc((void **)&d_ra_real, NoofReal * sizeof(float));
    cudaMalloc((void **)&d_dec_real, NoofReal * sizeof(float));
    cudaMalloc((void **)&d_ra_sim, NoofSim * sizeof(float));
    cudaMalloc((void **)&d_dec_sim, NoofSim * sizeof(float));

    // Allocate memory for the histograms on the GPU
    unsigned int *d_histogramDD, *d_histogramDR, *d_histogramRR;
    cudaMalloc((void **)&d_histogramDD, numBins * sizeof(unsigned int));
    cudaMalloc((void **)&d_histogramDR, numBins * sizeof(unsigned int));
    cudaMalloc((void **)&d_histogramRR, numBins * sizeof(unsigned int));

    // copy data to the GPU
    cudaMemcpyAsync(d_ra_real, ra_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_dec_real, decl_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_ra_sim, ra_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_dec_sim, decl_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // calculate total blocks and threads
    noofblocks = (NoofSim + threadsperblock - 1) / threadsperblock;
    printf("number of blocks: %d,\nnumber of threads in block: %d,\nnumber of total threads: %d\n", noofblocks, threadsperblock, threadsperblock * noofblocks);

    // Launch the kernels
    // DD
    calculateHistograms<<<noofblocks, threadsperblock>>>(d_ra_real, d_dec_real, NoofReal, d_ra_real, d_dec_real, NoofReal, d_histogramDD);

    // DR
    calculateHistograms<<<noofblocks, threadsperblock>>>(d_ra_real, d_dec_real, NoofReal, d_ra_sim, d_dec_sim, NoofSim, d_histogramDR);

    // RR
    calculateHistograms<<<noofblocks, threadsperblock>>>(d_ra_sim, d_dec_sim, NoofSim, d_ra_sim, d_dec_sim, NoofSim, d_histogramRR);

    // copy the results back to the CPU
    // Allocate memory on the host for the histograms
    histogramDD = (unsigned int *)malloc(numBins * sizeof(unsigned int));
    histogramDR = (unsigned int *)malloc(numBins * sizeof(unsigned int));
    histogramRR = (unsigned int *)malloc(numBins * sizeof(unsigned int));

    // Copy the histograms from device to host
    cudaMemcpyAsync(histogramDD, d_histogramDD, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(histogramDR, d_histogramDR, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(histogramRR, d_histogramRR, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(d_ra_real);
    cudaFree(d_dec_real);
    cudaFree(d_ra_sim);
    cudaFree(d_dec_sim);
    cudaFree(d_histogramDD);
    cudaFree(d_histogramDR);
    cudaFree(d_histogramRR);

    // Error handling
    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("    CUDA error: %s\n", cudaGetErrorString(myError));
        return (-1);
    }

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;
    kerneltime += end - start;

    printf("Total Kernel Execution Time: %f seconds\n", kerneltime);

    // calculate sum entries of histogramDRsum, histogramDDsum, histogramRRsum
    for (int i = 0; i < numBins; ++i)
        histogramDDsum += histogramDD[i];
    printf("Histogram DD sum = %lu\n", histogramDDsum);
    for (int i = 0; i < numBins; ++i)
        histogramDRsum += histogramDR[i];
    printf("Histogram DR sum = %lu\n", histogramDRsum);
    for (int i = 0; i < numBins; ++i)
        histogramRRsum += histogramRR[i];
    printf("Histogram RR sum = %lu\n", histogramRRsum);

    // calculate omega values on the CPU
    float *omega = (float *)malloc(numBins * sizeof(float));

    // Write results to file
    outfil = fopen("result.out", "w");
    if (outfil == NULL)
    {
        perror("Error opening file");
        free(omega);
        return -1;
    }

    for (int i = 0; i < numBins; i++)
    {
        if (histogramRR[i] != 0)
        {
            // omega calculation is based on the formula Wi(O) = (DDi - 2*DRi + RRi) / RRi
            omega[i] = (float)(histogramDD[i] - 2 * histogramDR[i] + histogramRR[i]) / histogramRR[i];
        }
        else
        {
            omega[i] = 0.0;
        }
        fprintf(outfil, "%d: %.3f, %.6f, %lu, %lu, %lu\n", i, (i * 0.25), omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
    }
    fclose(outfil);

    // Print only 5 omega values to out.txt | code above will save all omega values in result.out
    for (int i = 0; i < 5; i++)
    {
        printf("%d: %.3f, %.6f, %lu, %lu, %lu\n", i, (i * 0.25), omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
    }

    // Free CPU memory
    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    free(omega);

    return (0);
}

// spherical coordinates phi and theta:
// phi   = ra/60.0 * dpi/180.0;
float convertToRadian(float arcminutes)
{
    return arcminutes / 60.0 * (M_PI / 180.0);
}

int readdata(char *argv1, char *argv2)
{
    int i, linecount;
    char inbuf[180];
    double ra, dec, phi, theta, dpi;
    FILE *infil;

    printf("   Assuming input data is given in arc minutes!\n");
    // spherical coordinates phi and theta:
    // phi   = ra/60.0 * dpi/180.0;
    // theta = (90.0-dec/60.0)*dpi/180.0;

    dpi = acos(-1.0);
    infil = fopen(argv1, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv1);
        return (-1);
    }

    // read the number of galaxies in the input file
    int announcednumber;
    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv1);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 180, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv1, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
        return (-1);
    }

    NoofReal = linecount;
    ra_real = (float *)calloc(NoofReal, sizeof(float));
    decl_real = (float *)calloc(NoofReal, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv1);
            fclose(infil);
            return (-1);
        }
        ra_real[i] = (float)convertToRadian(ra);
        decl_real[i] = (float)convertToRadian(dec);
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv1);
        return (-1);
    }

    infil = fopen(argv2, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv2);
        return (-1);
    }

    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv2);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv2, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
        return (-1);
    }

    NoofSim = linecount;
    ra_sim = (float *)calloc(NoofSim, sizeof(float));
    decl_sim = (float *)calloc(NoofSim, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv2);
            fclose(infil);
            return (-1);
        }
        ra_sim[i] = (float)convertToRadian(ra);
        decl_sim[i] = (float)convertToRadian(dec);
        ++i;
    }

    fclose(infil);

    if (i != NoofSim)
    {
        printf("   Cannot read %s correctly\n", argv2);
        return (-1);
    }

    return (0);
}

int getDevice(int deviceNo)
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (-1);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");
        if (deviceProp.concurrentKernels == 1)
            printf("     yes\n");
        else
            printf("    no\n");
        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
            printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using CUDA device %d\n\n", device);

    return (0);
}
