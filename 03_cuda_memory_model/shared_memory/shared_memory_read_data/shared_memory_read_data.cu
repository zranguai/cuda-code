/**************************************************************************************
 *  file name   : shared_memory_read_data.cu
 *  brief       : 研究共享内存的数据分布，以及如何使用动态的共享内存，以及使用填充来避免冲突
 * 检测存储体冲突指标: shared_load_transactions_per_request/shared_store_transactions_per_request
 *  ref         : https://face2ai.com/CUDA-F-5-2-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E6%95%B0%E6%8D%AE%E5%B8%83%E5%B1%80/
***************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

// 测试方形共享内存
#define BDIMX 32
#define BDIMY 32

// 测试矩形共享内存
#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1  // +padding

__global__ void warmup(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}

// 这个核函数按照行主序进行读和写，所以对于共享内存没有读写冲突, 行主序的读和写事务是1
__global__ void setRowReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
// 这个核函数按照列主序进行读和写(所以线程都在一个存储体里面了)，对于共享内存有读写冲突, 行主序的读和写事务是32
__global__ void setColReadCol(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}

// 按行主序读列主序写
__global__ void setColReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;  // 列写
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];  // 行读
}
// 按行主序写列主序读
__global__ void setRowReadCol(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];  // 静态共享内存
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;  // 行写
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];  // 列读
}
__global__ void setRowReadColDyn(int * out)
{
    extern __shared__ int tile[];  // 动态共享内存
    unsigned int row_idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int col_idx=threadIdx.x*blockDim.y+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}

// 加IPAD进行填充， 填充后冲突都不见了！！
__global__ void setRowReadColIpad(int * out)
{
    __shared__ int tile[BDIMY][BDIMX+IPAD];  // 静态共享内存+padding 相当于在后面补充一列
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDynIpad(int * out)
{
    extern __shared__ int tile[];  // 动态共享内存+padding
    unsigned int row_idx=threadIdx.y*(blockDim.x+1)+threadIdx.x;  // 这里的blockDim.x+1, 因为加了padding,相当于每一行多一个元素
    unsigned int col_idx=threadIdx.x*(blockDim.x+1)+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}


//---------------------------rectagle(需要先转换成线性然后再重新计算行和列的坐标)----------------------------

// 行读和行写
__global__ void setRowReadColRect(int * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;  // 这里并没有改变元素tile形状，只是改变了数组的索引顺序
    unsigned int irow=idx/blockDim.y;  // 通过除法和取余 逐列进行。这样就不是前面的32个冲突而是16个
    tile[threadIdx.y][threadIdx.x]=idx;  // 行写
    __syncthreads();
    out[idx]=tile[icol][irow];  // 行读
}
__global__ void setRowReadColRectDyn(int * out)
{
    extern __shared__ int tile[];  // 动态共享内存
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    unsigned int col_idx=icol*blockDim.x+irow;
    tile[idx]=idx;
    __syncthreads();
    out[idx]=tile[col_idx];
}

// +padding
__global__ void setRowReadColRectPad(int * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT+IPAD*2];  // 这里填充一列时会产生两路冲突，填充两列时就没有冲突，所以这里填充两列
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[icol][irow];
}
__global__ void setRowReadColRectDynPad(int * out)
{
    extern __shared__ int tile[];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    unsigned int row_idx=threadIdx.y*(IPAD+blockDim.x)+threadIdx.x;  // 这里也是加一列 有两路冲突
    unsigned int col_idx=icol*(IPAD+blockDim.x)+irow;
    tile[row_idx]=idx;
    __syncthreads();
    out[idx]=tile[col_idx];
}


int main(int argc,char **argv)
{
  // set up device
  initDevice(0);
  int kernel=0;
  if(argc>=2)
    kernel=atoi(argv[1]);
  int nElem=BDIMX*BDIMY;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(int)*nElem;
  int * out;
  CHECK(cudaMalloc((int**)&out,nByte));

  cudaSharedMemConfig MemConfig;
  CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));
  printf("--------------------------------------------\n");
  switch (MemConfig) {  // 判断共享内存的存储体宽度

      case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n");
      break;
      case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n");
      break;

  }

  printf("--------------------------------------------\n");
  dim3 block(BDIMY,BDIMX);  // BDIMY,BDIMX均为32
  dim3 grid(1,1);

  dim3 block_rect(BDIMX_RECT,BDIMY_RECT);
  dim3 grid_rect(1,1);

  warmup<<<grid,block>>>(out);
  printf("warmup!\n");
  double iStart,iElaps;
  iStart=cpuSecond();
  switch(kernel)
  {
      case 0:
          {
          setRowReadRow<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadRow  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
      //break;
      //case 1:
          iStart=cpuSecond();
          setColReadCol<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setColReadCol  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
        }
      case 2:
        {
          setColReadRow<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setColReadRow  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
        }
      case 3:
      {
          setRowReadCol<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadCol  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 4:
      {
            // 动态共享内存需要设置第三个参数 (BDIMX)*BDIMY*sizeof(int) 为数组大小
            setRowReadColDyn<<<grid,block,(BDIMX)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();
            iElaps=cpuSecond()-iStart;
            printf("setRowReadColDyn  ");
            printf("Execution Time elapsed %f sec\n",iElaps);
            break;
        }
      case 5:
      {
          setRowReadColIpad<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColIpad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 6:
      {
          // 动态共享内存+padding, 申请空间需要加上IPAD即在最后加上一列
          setRowReadColDynIpad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColDynIpad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 7:
      {
          setRowReadColRect<<<grid_rect,block_rect>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRect  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 8:
      {
          setRowReadColRectDyn<<<grid_rect,block_rect,(BDIMX)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectDyn  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 9:
      {
          setRowReadColRectPad<<<grid_rect,block_rect>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectPad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 10:
      {
          setRowReadColRectDynPad<<<grid_rect,block_rect,(BDIMX+1)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectDynPad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 11:
      {
            setRowReadRow<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setColReadCol<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setColReadRow<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadCol<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadColDyn<<<grid,block,(BDIMX)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();

            setRowReadColIpad<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadColDynIpad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();
            break;
    }
    case 12:
    {
        setRowReadColRect<<<grid_rect,block_rect>>>(out);
        setRowReadColRectDyn<<<grid_rect,block_rect,(BDIMX)*BDIMY*sizeof(int)>>>(out);
        setRowReadColRectPad<<<grid_rect,block_rect>>>(out);
        setRowReadColRectDynPad<<<grid_rect,block_rect,(BDIMX+1)*BDIMY*sizeof(int)>>>(out);
        break;
    }

  }

  cudaFree(out);
  return 0;
}