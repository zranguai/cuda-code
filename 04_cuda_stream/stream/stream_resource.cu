/**************************************************************************************
 *  brief       : 观察Hyper-Q工作队列一级内核资源占用对并行程度的影响
***************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define N 100
__global__ void kernel_1()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_2()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_3()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_4()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
int main()
{
    //setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);  // // 这里设置Hyper-Q的工作队列
    int n_stream=4;
    cudaStream_t *stream=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
    for(int i=0;i<n_stream;i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    // 当内核的线程数目增加的时候，内核级别的并行数量就会下降
    // dim3 block(1);
    // dim3 grid(1);
    // 例如升级到下面的时候，并行度就会减少
    dim3 block(16,32);
    dim3 grid(32);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<n_stream;i++)
    {
        kernel_1<<<grid,block,0,stream[i]>>>();
        kernel_2<<<grid,block,0,stream[i]>>>();
        kernel_3<<<grid,block,0,stream[i]>>>();
        kernel_4<<<grid,block,0,stream[i]>>>();
    }
    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    printf("elapsed time:%f ms\n",elapsed_time);

    for(int i=0;i<n_stream;i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(stream);
    return 0;
}
