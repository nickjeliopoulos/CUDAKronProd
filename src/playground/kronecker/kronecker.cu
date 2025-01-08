#include "kronecker.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::kronecker {
	namespace{
		using uint = unsigned int;

		struct problem_size_t{
			uint MA, NA;
			uint MB, NB;
			uint MC, NC;
			uint num_stages;
			uint num_stages_remainder;
		};

		struct __builtin_align__(8) dim2{
			uint x, y;
			dim2(uint x, uint y) : x(x), y(y) {}
		};

		constexpr uint SM80_THREADS_PER_WARP = 32;
		constexpr uint SM80_KRONECKER_PROBLEM_THREADS = 512;
		constexpr uint SM80_KRONECKER_PROBLEM_WARPS = SM80_KRONECKER_PROBLEM_THREADS / SM80_THREADS_PER_WARP;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS = 4;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS = SM80_KRONECKER_PROBLEM_THREADS;
		constexpr uint SYNCWARP_ALL_MASK = 0b11111111111111111111111111111111;

		void validate_input(const torch::Tensor& A, const torch::Tensor& B){
			TORCH_CHECK(A.dim() == 2, "A must be a 2D Tensor");
			TORCH_CHECK(B.dim() == 2, "B must be a 2D Tensor");
			TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
			TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
			TORCH_CHECK(A.get_device() == B.get_device(), "Tensors A and B must be on the same device");
		}

		std::string stringify_workload_information(const char* function_name, const dim3& threadblocks, const dim3& threads){
			char workload_info_str[128];
			sprintf_s(workload_info_str, "%s: Thread Block <%d,%d,%d> | Threads <%d,%d,%d>\n", function_name, threadblocks.x, threadblocks.y, threadblocks.z, threads.x, threads.y, threads.z);
			return std::string(workload_info_str);
		}
	}


	problem_size_t get_kronecker_problem_size(const torch::Tensor& A, const torch::Tensor& B){
		problem_size_t problem_size;
		problem_size.MA = A.size(0);
		problem_size.NA = A.size(1);
		problem_size.MB = B.size(0);
		problem_size.NB = B.size(1);
		problem_size.MC = problem_size.MA * problem_size.MB;
		problem_size.NC = problem_size.NA * problem_size.NB;
		problem_size.num_stages = problem_size.MB / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS;
		problem_size.num_stages_remainder = problem_size.MB % SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS;
		assert(problem_size.num_stages > 0);
		return problem_size;
	}
	

   template <typename scalar_t>
	__global__ void kronecker_anyrow_anycol_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C,
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC,
		const uint column_stages){
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint stage_id = blockIdx.z;
		const uint tx = threadIdx.x;

		uint column_idx = 0;

		// TODO: Adapt 
		// __shared__ float smem_B[SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS][SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS];
		// const float AIJ = A[I][J];

		// // Prefill SMEM with chunk of B
		// #pragma unroll
		// for(uint k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
		// 	// Load B into SMEM
		// 	smem_B[k][tx] = B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][tx];
		// }
		// __syncthreads();

		// Prefill SMEM with chunk of B
		#pragma unroll
		for(uint k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			for(uint l = 0; l < column_stages; l++){
				column_idx = l*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS + tx;
				if(column_idx < NB) C[I*MA + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NA + column_idx] = A[I][J] * B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][column_idx];
			}
		}
	}

	//
	// Core Operators
	// Each of these operators is designed to handle a set of specific problems sizes OR has some type of alternative implementation (naive, or optimized for some type of metric)
	//
	torch::Tensor kronecker_anyrow_anycol_product(const torch::Tensor& A, const torch::Tensor& B){	
		validate_input(A, B);

		// Get Problem Size + Initialize C
		const problem_size_t problem_size = get_kronecker_problem_size(A, B);
		torch::Tensor C = torch::empty({problem_size.MC, problem_size.NC}, A.options());

		// CUDA Kernel Launch:
		// Set Kernel Launch Parameters + Launch
		const dim3 threadblocks(problem_size.MA, problem_size.NA, problem_size.num_stages);
		const dim3 threads(SM80_KRONECKER_PROBLEM_THREADS);

		std::cout << stringify_workload_information(__func__, threadblocks, threads);

		kronecker_anyrow_anycol_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
			A.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			B.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			problem_size.MA, 
			problem_size.NA, 
			problem_size.MB, 
			problem_size.NB, 
			problem_size.MC, 
			problem_size.NC,
			// Replacement for: (uint)ceil( (float)NB / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS)
			(problem_size.NB + SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS - 1U) / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS
		); 

		return C;
	}


	//
	// Adapative Kronecker Operator - invoke the corresponding Core Operator based on problem size
	//
	torch::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B){	
		validate_input(A, B);

		// Get Problem Size + Initialize C
		const problem_size_t problem_size = get_kronecker_problem_size(A, B);
		torch::Tensor C = torch::empty({problem_size.MC, problem_size.NC}, A.options());

		// CUDA Kernel Launch:
		// Set Kernel Launch Parameters + Launch
		const dim3 threadblocks(problem_size.MA, problem_size.NA, problem_size.num_stages);
		const dim3 threads(SM80_KRONECKER_PROBLEM_THREADS);

		std::cout << stringify_workload_information(__func__, threadblocks, threads);

		kronecker_tiny_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
			A.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			B.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
			problem_size.MA, 
			problem_size.NA, 
			problem_size.MB, 
			problem_size.NB, 
			problem_size.MC, 
			problem_size.NC
		); 

		return C;
	}
}
