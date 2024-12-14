#include "kronecker.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::kronecker {
	namespace{
		using uint = unsigned int;

		struct __builtin_align__(32) problem_size_t{
			uint MA, NA;
			uint MB, NB;
			uint MC, NC;
		};

		constexpr uint SM80_THREADS_PER_WARP = 32;
		constexpr uint SM80_MAX_THREADS = 1536;
		constexpr uint SM80_MAX_WARPS = SM80_MAX_THREADS / SM80_THREADS_PER_WARP;
		constexpr uint SM80_KRONECKER_PROBLEM_THREADS = 256;
	}

	problem_size_t get_kronecker_problem_size(const torch::Tensor& A, const torch::Tensor& B){
		problem_size_t problem_size;
		problem_size.MA = A.size(0);
		problem_size.NA = A.size(1);
		problem_size.MB = B.size(0);
		problem_size.NB = B.size(1);
		problem_size.MC = problem_size.MA * problem_size.MB;
		problem_size.NC = problem_size.NA * problem_size.NB;
		return problem_size;
	}


	// Kernels for Kronecker Product
	// Each threadblock is responsible for a chunk of C, A[i][j] * B
	// Each thread computes a A[i][j] * B[k][l] 
	// ASSUMPTION #1: MB*NB <= SM80_KRONECKER_PROBLEM_THREADS
	// TODO: Solve this constraint, it should be easy with a simple for loop.
    template <typename scalar_t>
	__global__ void _kronecker_sm80_bf16_bf16_cuda_kernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC){
		// Thread Indexing
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint tx = threadIdx.x;

		// Shared Memory and Variable Init.
		__shared__ float smem_B[16][16];
		float A_IJ = A[I][J];

		// Load elements of B into Shared Memory
		// We can one-shot this if tx < MB * NB <= SM80_KRONECKER_PROBLEM_THREADS
		if(tx < MB * NB){
			const uint k = tx / NB;
			const uint l = tx % NB;
			smem_B[k][l] = B[k][l];
		}
		__syncthreads();

		// Compute C
		for(uint k = 0; k < MB; k++){
			for(uint l = 0; l < NB; l++){
				C[I*MB + k][J*NB + l] = A_IJ * smem_B[k][l];
			}
		}
		// Epilogue 
	}


	// Operator for Kronecker Product
	at::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B){	
		// Input Checking
		// TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be of type BF16");
		// TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "A must be of type BF16");
		TORCH_CHECK(A.dim() == 2, "A must be a 2D Tensor");
		TORCH_CHECK(B.dim() == 2, "B must be a 2D Tensor");
		TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
		TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
		TORCH_CHECK(A.get_device() == B.get_device(), "Tensors A and B must be on the same device");

		// Get Problem Size + Initialize C
		const problem_size_t problem_size = get_kronecker_problem_size(A, B);
		torch::Tensor C = torch::empty({problem_size.MC, problem_size.NC}, A.options());

		// CUDA Kernel Launch:
		// Set Kernel Launch Parameters + Launch
		const dim3 threadblocks(problem_size.MA, problem_size.NA);
		const dim3 threads(SM80_KRONECKER_PROBLEM_THREADS);

		// Torch Dispatch
		// TODO: Fix
		// AT_DISPATCH_FLOATING_TYPES(torch::kFloat32, "gorby_kronecker", ([&]{ 
		// 	_kronecker_sm80_bf16_bf16_cuda_kernel<float><<<threadblocks, threads>>>(
		// 		A.packed_accessor32<torch::kFloat32, 2, torch::RestrictPtrTraits>(),
		// 		B.packed_accessor32<torch::kFloat32, 2, torch::RestrictPtrTraits>(),
		// 		C.packed_accessor32<torch::kFloat32, 2, torch::RestrictPtrTraits>(),
		// 		problem_size.MA, 
		// 		problem_size.NA, 
		// 		problem_size.MB, 
		// 		problem_size.NB, 
		// 		problem_size.MC, 
		// 		problem_size.NC
		// 	); 
		// }));

		// Vanilla CUDA/C++ Kernel Launch
		// _kronecker_sm80_bf16_bf16_cuda_kernel<float><<<threadblocks, threads>>>(
		// 	A.packed_accessor32<torch::kFloat32, 2, at::RestrictPtrTraits>(), 
		// 	B.packed_accessor32<torch::kBFloat16, 2, at::RestrictPtrTraits>(), 
		// 	C.packed_accessor32<torch::kBFloat16, 2, at::RestrictPtrTraits>(), 
		// 	problem_size.MA, 
		// 	problem_size.NA, 
		// 	problem_size.MB, 
		// 	problem_size.NB, 
		// 	problem_size.MC, 
		// 	problem_size.NC
		// );

		return C;
	}
}
