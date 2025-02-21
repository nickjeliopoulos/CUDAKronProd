#include <cuda_runtime.h>
#include <iostream>


__device__ float sigmoid(float x, float beta) {
	return 1.0f / fmaf(expf(-x), beta, 1.0f);
}


__global__ void sigmoid_kernel(float* input, float* output, float beta){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	output[idx] = sigmoid(input[idx], beta);
}


__global__ void swiglu_kernel(float* y, float* x, float* W, float* V, float b, float c, float beta) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	y[idx] = x[idx] * sigmoid(x[idx], beta);
}

void swiglu(float* y, float* x, float* W, float* V, float b, float c, float beta, int n){
	float *d_y, *d_x, *d_W, *d_V;
	
	cudaMalloc(&d_y, n * sizeof(float));
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_W, n * sizeof(float));
	cudaMalloc(&d_V, n * sizeof(float));

	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice);

	swiglu_kernel<<<n/256, 256>>>(d_y, d_x, d_W, d_V, b, c, beta);

	cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_y);
	cudaFree(d_x);
	cudaFree(d_W);
	cudaFree(d_V);
}


int main(){
	return 0;
}