#include "swiglu.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::swiglu{
	namespace{
		using uint = unsigned int;

		struct problem_size_t{
			
		};

	}	


	problem_size_t get_swiglu_problem_size(const torch::Tensor& A, const torch::Tensor& B){

	}


	//
	// Need Sigmoid for Swish
	//
	__device__ float sigmoid(float x, float beta) {
		return 1.0f / fmaf(expf(-x), beta, 1.0f);
	}


	//
	// Swish activation function
	//
	__device__ float swish(float x, float y, float beta) {
		return fmaf(x, sigmoid(y, beta), 0.0f);
	}


	// Kernel Grid Size: 
	// Kernel Thread Size: 
	template <typename scalar_t>
	__global__ void swiglu_2d_fp32_cuda_kernel(

	) {

	}


	// SWiGLU Operator
    torch::Tensor swiglu(const torch::Tensor& A, const torch::Tensor& B) {
		return A
	}

}