//一維陣列相加的範例程式

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// #define DataSize 16
#define DataSize 64

void GenerateNumbers(int *number, int size, int k)//隨機產生資料
{
     int i;
	   srand(k * time(NULL));
     for (i = 0; i < size; i++)
         number[i] = rand() % 100;
}

__global__ void Add(int *Da, int *Db, int *Dc)//kernel function
{
	int tx = threadIdx.x;           //thread的x軸id
	int bx = blockIdx.x;			//block的x軸id
	int bn = blockDim.x;			//block的x軸有幾個thread
	int id = bx*bn+tx;				//計算矩陣座標

	// Dc[id] = Da[id] + Db[id];
	for (int i = 0; i < DataSize; i += 16)
	{
		Dc[id + i] = Da[id + i] + Db[id + i];
	}
}

int main()
{
	int *Ha, *Hb, *Hc; //CPU
	int size = DataSize * sizeof(int);
	
	Ha = (int*)malloc(size);				//配置矩陣空間
	Hb = (int*)malloc(size);				//配置矩陣空間
	Hc = (int*)malloc(size);				//配置矩陣空間
	
	GenerateNumbers(Ha, DataSize, 2);		//產生矩陣資料
	GenerateNumbers(Hb, DataSize, 6);		//產生矩陣資料

	
	/* dim3 由CUDA提供的三維向量型態 (X,Y,Z)
		CUDA限制每個block的thread上限為1024, (X*Y*Z)<=1024
		grid的block上限為65535, (X*Y)<=65535.  block最多2維而已
	*/	
	// dim3 block(DataSize/2, 1, 1);			//配置 block 內 thread維度、每個維度的大小 
	dim3 block(DataSize/8, 1, 1);			//配置 block 內 thread維度、每個維度的大小 
	dim3 grid(2, 1, 1);						//配置 grid 內 block維度、每個維度的大小

	
	
	int *Da, *Db, *Dc; //GPU
	cudaMalloc((void**)&Da, size);			//配置GPU矩陣空間
	cudaMalloc((void**)&Db, size);			//配置GPU矩陣空間
	cudaMalloc((void**)&Dc, size);			//配置GPU矩陣空間

	cudaMemcpy(Da, Ha, size, cudaMemcpyHostToDevice);		//複製資料到GPU
	cudaMemcpy(Db, Hb, size, cudaMemcpyHostToDevice);		//複製資料到GPU

	Add <<< grid, block >>> (Da, Db, Dc);			//呼叫kernel
	cudaThreadSynchronize();                // ??

	cudaMemcpy(Hc, Dc, size, cudaMemcpyDeviceToHost);		//複製資料(相加後的結果)回CPU

	int i;
	printf("A {\n");
	for (i = 0; i < DataSize; i++)
		printf("\"No.%2d\" : %3d, ", i+1, Ha[i]);
	printf("}\n");
	
	printf("B {\n");
	for (i = 0; i < DataSize; i++)
		printf("\"No.%2d\" : %3d, ", i+1, Hb[i]);
	printf("}\n");
	
	printf("C\n");
	for (i = 0; i < DataSize; i++)
		printf("\"No.%2d\" : %3d, ", i+1, Hc[i]);
	printf("}\n");

	
	//釋放記憶體空間
	free(Ha); free(Hb); free(Hc);
	cudaFree(Da); cudaFree(Db); cudaFree(Dc);
}
