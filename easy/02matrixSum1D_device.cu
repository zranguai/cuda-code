/*********************************************************************************************
 * file name  : matrixSum1Ddevice.cu
 * brief      : 矩阵求和程序，通过调用核函数在GPU执行
 *              添加：1、if判断条件；2、调用设备函数
***********************************************************************************************/

#include <stdio.h>
#include "include/common.cuh"

// __device__ device调用，device执行
__device__ float add(const float x, const float y)
{
    return x + y;
}

// __global__ host调用，device执行
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; // 513  32*17=544

    if (id >= N) return;  // 因为id有可能大于N
    C[id] = add(A[id], B[id]);
    
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int iElemCount = 513;                     // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float); // 字节数
    
    // （1）分配主机内存，并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }


    // （2）分配设备内存，并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);  // 这里用双指针，因为要传递值
    // cuda错误检查， __FILE__显示的是啥文件，__LINE__显示的是该行
    cudaError error = ErrorCheck(cudaMalloc((float**)&fpDevice_B, stBytesCount), __FILE__, __LINE__);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);
    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);  // 设备内存初始化为0
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        printf("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 3、初始化主机中数据
    srand(666); // 设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);
    
    // 4、数据从主机复制到设备
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);


    // 5、调用核函数在设备中进行计算
    dim3 block(32);
    // dim3 grid(iElemCount / 32);  // 513 / 32 = 16
    dim3 grid((iElemCount + block.x - 1) / 32); //17

    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);    // 调用核函数

    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);  // 捕捉核函数错误
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // cudaDeviceSynchronize();  // 等待kernel执行完

    // 6、将计算得到的数据从设备传给主机， 这种方式有隐式同步，即cudaDeviceSynchronize()可以不用
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);


    for (int i = 0; i < 10; i++)    // 打印
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 7、释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}