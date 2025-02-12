#include <cuda_runtime.h>
#include <iostream>


__device__ float sigmoid(float x, float beta) {
	return 1.0f / fmaf(expf(-x), beta, 1.0f);
}


__global__ void sigmoid_kernel(float* input, float* output, float beta){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	output[idx] = sigmoid(input[idx], beta);
}


int main(){
	return 0;
}