/**************************************************************************************
 *  file name   : transform_matrix2D.cu
 *  brief       : 使用矩阵转置研究内存带宽利用率
 *  ref         : https://face2ai.com/CUDA-F-4-4-%E6%A0%B8%E5%87%BD%E6%95%B0%E5%8F%AF%E8%BE%BE%E5%88%B0%E7%9A%84%E5%B8%A6%E5%AE%BD/
***************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

// cpu实现矩阵转置
void transformMatrix2D_CPU(float * MatA,float * MatB,int nx,int ny)
{
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      MatB[i*nx+j]=MatA[j*nx+i];
    }
  }
}

// 行读取，行存储 来复制矩阵(测试带宽利用上限) 能够合并访问
__global__ void copyRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
// 列读取，列存储 来复制矩阵(测试内存带宽利用下限)
__global__ void copyCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
// 使用命令行编译，开启一级缓存
// nvcc -O3 -arch=sm_35 -Xptxas -dlcm=ca -I ../include/ transform_matrix2D.cu -o transform_matrix2D


// 朴素的转置 行读取列存储
__global__ void transformNaiveRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
    }
}
// 列读取 行存储
__global__ void transformNaiveCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_row]=MatA[idx_col];
    }
}
// 这里按列读取效果更好 因为能够利用到缓存


// 展开转置，有效地隐藏延迟
__global__ void transformNaiveRowUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
      MatB[idx_col+ny*1*blockDim.x]=MatA[idx_row+1*blockDim.x];
      MatB[idx_col+ny*2*blockDim.x]=MatA[idx_row+2*blockDim.x];
      MatB[idx_col+ny*3*blockDim.x]=MatA[idx_row+3*blockDim.x];
    }
}
__global__ void transformNaiveColUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
        MatB[idx_row]=MatA[idx_col];
        MatB[idx_row+1*blockDim.x]=MatA[idx_col+ny*1*blockDim.x];
        MatB[idx_row+2*blockDim.x]=MatA[idx_col+ny*2*blockDim.x];
        MatB[idx_row+3*blockDim.x]=MatA[idx_col+ny*3*blockDim.x];
    }
}


// 利用DRAM的特性：DRAM中内存是分区规划的，如果过多的访问同一个区，会产生排队的现象，
// 也就是要等待，为了避免这种情况，我们最好均匀的访问DRAM的某一段
__global__ void transformNaiveRowDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;  // 这里打乱为了利用DRAM特性
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];  // 这里的idx_row其实和数组的index其实是对应这的
    }
}
__global__ void transformNaiveColDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_row]=MatA[idx_col];
    }
}

int main(int argc,char** argv)
{
  printf("strating...\n");
  initDevice(0);
  int nx=1<<12;
  int ny=1<<12;
  int dimx=32;
  int dimy=32;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);
  int transform_kernel=0;
  if(argc==2)
    transform_kernel=atoi(argv[1]);
  if(argc>=4)
  {
      transform_kernel=atoi(argv[1]);
      dimx=atoi(argv[2]);
      dimy=atoi(argv[3]);
  }

  //Malloc
  float* A_host=(float*)malloc(nBytes);
  float* B_host=(float*)malloc(nBytes);
  initialData(A_host,nxy);

  //cudaMalloc
  float *A_dev=NULL;
  float *B_dev=NULL;
  CHECK(cudaMalloc((void**)&A_dev,nBytes));
  CHECK(cudaMalloc((void**)&B_dev,nBytes));

  CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(B_dev,0,nBytes));



  // cpu compute
  double iStart=cpuSecond();
  transformMatrix2D_CPU(A_host,B_host,nx,ny);
  double iElaps=cpuSecond()-iStart;
  printf("CPU Execution Time elapsed %f sec\n",iElaps);

  // 2d block and 2d grid
  dim3 block(dimx,dimy);
  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
  dim3 block_1(dimx,dimy);
  dim3 grid_1((nx-1)/(block_1.x*4)+1,(ny-1)/block_1.y+1);
  iStart=cpuSecond();
  switch(transform_kernel)
  {
  case 0:
    copyRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 1:
    copyCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 2:
    transformNaiveRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 3:
        transformNaiveCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  case 4:
        transformNaiveColUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
        break;
  case 5:

        transformNaiveColUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
        break;
  case 6:
        transformNaiveRowDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  case 7:
        transformNaiveColDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  default:
    break;
  }
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf(" Time elapsed %f sec\n",iElaps);
  CHECK(cudaMemcpy(B_host,B_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(B_host,B_host,nxy);

  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A_host);
  free(B_host);
  cudaDeviceReset();
  return 0;
}