#define N 16
#include <stdio.h>

__global__ void matrixMult (int *a, int *b, int *c, int width);

int main() {
	int a[N][N], b[N][N], c[N][N];
	int *dev_a, *dev_b, *dev_c;

	// initialize matrices a and b with appropriate values
	for (int i = 0; i < N; ++i)
	 {
	 	for (int j = 0; j < N; ++j)
	 	{
	 		a[i][j] = rand() % 10;
	 		b[i][j] = rand() % 10;
	 	}
	 } 
	
	int size = N * N * sizeof(int);
	
	cudaMalloc((void **) &dev_a, size);
	cudaMalloc((void **) &dev_b, size);
	cudaMalloc((void **) &dev_c, size);
	
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(1, 1);
	dim3 dimBlock(N, N);
	
	matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
	
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

	int i, j;
	printf("a {\n");
	for (i = 0; i < N; i++) 
	{
		for (j = 0; j < N; ++j)
		{
			printf("\"[%2d][%2d]\" : %3d, ", i, j, a[i][j]);	
		}
	}
	printf("}\n");

	printf("b {\n");
	for (i = 0; i < N; i++) 
	{
		for (j = 0; j < N; ++j)
		{
			printf("\"[%2d][%2d]\" : %3d, ", i, j, b[i][j]);	
		}
	}
	printf("}\n");

	printf("c {\n");
	for (i = 0; i < N; i++) 
	{
		for (j = 0; j < N; ++j)
		{
			printf("\"[%2d][%2d]\" : %3d, ", i, j, c[i][j]);	
		}
	}
	printf("}\n");
}

__global__ void matrixMult (int *a, int *b, int *c, int width) {
	int k, sum = 0;
	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	
	if(col < width && row < width) {
 		for (k = 0; k < width; k++) {
			sum += a[k * width + row] * b[col * width + k];
			c[col * width + row] = sum;
 		}
	}
}