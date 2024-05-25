
/**************************************************************************************
 *  file name   : static_shared_memory.cu
 *  brief       : 静态共享内存使用
***************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__global__ void kernel_1(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ float s_array[32];
    // 静态共享内存可支持二维数组
    // size_x = 32; size_y= 32;
    // __shared__ float a[size_x][size_y]  // size_x, size_y需要编译时确定不能是变量

    if (n < N)
    {
        s_array[tid] = d_A[n];
    }
    __syncthreads();  // 线程同步

    if (tid == 0)
    {
        for (int i = 0; i < 32; ++i)
        {
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }

}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int nElems = 64;
    int nbytes = nElems * sizeof(float);

    float* h_A = nullptr;
    h_A = (float*)malloc(nbytes);
    for (int i = 0; i < nElems; ++i)
    {
        h_A[i] = float(i);
    }

    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes,cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);
    kernel_1<<<grid, block>>>(d_A, nElems);

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());

}