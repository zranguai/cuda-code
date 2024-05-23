/*********************************************************************************************
 * file name  :global_variable.cu
 * brief      : 静态全局变量使用
 * 1.在主机端，devData只是一个标识符，不是设备全局内存的变量地址
 * 2.在核函数中，devData就是一个全局内存中的变量。
***********************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
__device__ float devData;
__global__ void checkGlobalVariable()
{
    printf("Device: The value of the global variable is %f\n",devData);
    devData+=2.0;
}
int main()
{
    float value=3.14f;
    cudaMemcpyToSymbol(devData,&value,sizeof(float));
    printf("Host: copy %f to the global variable\n",value);
    checkGlobalVariable<<<1,1>>>();
    cudaMemcpyFromSymbol(&value,devData,sizeof(float));
    printf("Host: the value changed by the kernel to %f \n",value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}