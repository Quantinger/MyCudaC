#include <cuda_runtime.h>
#include <stdio.h>

/*
cudaError_t cudaDeviceSynchronize(void);

typedef union cudaError {
    cudaSuccess,
    cudaErrorMemoryAllocation,
}cudaError_t;

char *cudaGetErrorString(cudaError_t error);
*/
#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
        printf("code: %d, reason: %s", error, cudaGetErrorString(error));        \
        exit(1);                                                                 \
    }                                                                            \
}

/*
CHECK(cudaMemcpy(d_C, gpuRef, cudaMemcpyDeviceToHost));

OR

kernel_function <<<grid, block>>> (arguList);
CHECK(cudaDeviceSynchronize());
*/


void initialData(float *ip, int size) 
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF)/10.0f;
    }
}


/*
cudaError_t cudaMalloc(void **devPtr, size_t size);

union cudaMemcpyKind {
cudaMemcpyHostToDevice,
cudaMemcpyHostToHost,
cudaMemcpyDeviceToHost,
cudaMemcpyDeviceToDevice
};

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

char *cudaGetErrorString(cudaError_t error);

cudaFree();


__global__ void func(argList){
    //TODO
}

dim3 block(x, y, z);
dim3 grid((nElem + block.x - 1)/block.x);

func <<<block, grid>>> (argList);
*/

__global__ void sumArraysOnGPU(float *A, float *B, float *C) 
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) 
{
    for (int i = 0; i < N; i ++) {
        C[i] = A[i] + B[i];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N) 
{
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < N; i ++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);

            break;
        }
    }

    if (match) {
        printf("Arrays match.\n");
    }
}


int main(int argc, char **argv) 
{
    printf("%s Starting... \n");

    int dev = 0;
    cudaSetDevice(dev);

    // Set up data size of vectors
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hhostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArraysOnGPU <<<block, grid>>> (d_A, d_B, d_C);

    cudaMemcpy(d_C, gpuRef, cudaMemcpyDeviceToHost);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}
