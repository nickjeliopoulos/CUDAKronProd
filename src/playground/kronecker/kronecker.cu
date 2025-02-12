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
			uint batched;
		};

		constexpr uint SM80_THREADS_PER_WARP = 32;
		constexpr uint SM80_KRONECKER_PROBLEM_THREADS = 128;
		constexpr uint SM80_KRONECKER_PROBLEM_WARPS = SM80_KRONECKER_PROBLEM_THREADS / SM80_THREADS_PER_WARP;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS = 1;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS = SM80_KRONECKER_PROBLEM_THREADS;

		void validate_input(const torch::Tensor& A, const torch::Tensor& B){
			TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D/3D Tensor");
			TORCH_CHECK(B.dim() == 2 || B.dim() == 3, "B must be a 2D/3D Tensor");
			TORCH_CHECK(A.dim() == B.dim(), "A and B must have the same number of dims");
			TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
			TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
			TORCH_CHECK(A.get_device() == B.get_device(), "Tensors A and B must be on the same device");
		}

		std::string stringify_workload_information(const char* function_name, const problem_size_t& probelm_size, const dim3& threadblocks, const dim3& threads){
			char workload_info_str[128];
			sprintf_s(workload_info_str, 
				"%s: Thread Block <%d,%d,%d> | Threads <%d,%d,%d> | Batched: %s\n", 
				function_name, 
				threadblocks.x, 
				threadblocks.y, 
				threadblocks.z, 
				threads.x, 
				threads.y, 
				threads.z, 
				probelm_size.batched ? "True" : "False"
			);
			return std::string(workload_info_str);
		}
	}


	problem_size_t get_kronecker_problem_size(const torch::Tensor& A, const torch::Tensor& B){
		problem_size_t problem_size;

		// Identify whether we have a batched problem or not
		problem_size.batched = A.dim() > 2;

		// Set input workload size, based on batched or non-batched problem
		if(problem_size.batched){
			problem_size.MA = A.size(1);
			problem_size.NA = A.size(2);
			problem_size.MB = B.size(1);
			problem_size.NB = B.size(2);
		}
		else{
			problem_size.MA = A.size(0);
			problem_size.NA = A.size(1);
			problem_size.MB = B.size(0);
			problem_size.NB = B.size(1);
		}
		
		// Compute output workload size
		problem_size.MC = problem_size.MA * problem_size.MB;
		problem_size.NC = problem_size.NA * problem_size.NB;

		// Kernel computation parameters
		problem_size.num_stages = problem_size.MB / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS;
		problem_size.num_stages_remainder = problem_size.MB % SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS;

		// Requirement for current implementation. Address later
		assert(problem_size.num_stages > 0);

		return problem_size;
	}
	

	// Kernel Grid Size: (B * MA, NA, num_stages)
	// Kernel Thread Size: (SM80_KRONECKER_PROBLEM_THREADS)
	template <typename scalar_t>
	__global__ void kronecker_3d_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> C,
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC,
		const uint column_stages){
		// Get Thread Index
		const uint batch_id = blockIdx.x / MA;
		const uint I = blockIdx.x % MA;
		const uint J = blockIdx.y;
		const uint stage_id = blockIdx.z;
		const uint tx = threadIdx.x;	

		// Shared memory, thread constants
		const float AIJ = A[batch_id][I][J];

		// Loop variables
		uint column_idx = 0;

		#pragma unroll
		for(int32_t k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			// Handle all-but-last column chunk of C
			for(int32_t l = 0; l < column_stages-1; l++){
				column_idx = l*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS + tx;
				C[batch_id][I*MB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + column_idx] = __fmaf_rz(AIJ, B[batch_id][stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][column_idx], 0.0f);
			}

			// Handle last column chunk of C
			// Need to check a condition to avoid out-of-bounds access
			column_idx = (column_stages-1)*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS + tx;
			if(column_idx < NB){
				C[batch_id][I*MB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + column_idx] = __fmaf_rz(AIJ, B[batch_id] [stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][column_idx], 0.0f);
			}
		}
	}


	// Kernel Grid Size: (MA, NA, num_stages)
	// Kernel Thread Size: (SM80_KRONECKER_PROBLEM_THREADS)
	template <typename scalar_t>
	__global__ void kronecker_2d_sm80_fp32_fp32_cuda_kernel(
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
		// Get Thread Index
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint stage_id = blockIdx.z;
		const uint tx = threadIdx.x;

		// Shared Memory, thread constants
		const float AIJ = A[I][J];

		// Loop Variables (can help nvcc with unrolling loops)
		uint column_idx = 0;
	
		#pragma unroll
		for(int32_t k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			// Handle all-but-last column chunk of C
			for(int32_t l = 0; l < column_stages-1; l++){
				column_idx = l*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS + tx;
				C[I*MB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + column_idx] = __fmaf_rz(AIJ, B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][column_idx], 0.0f);
			}

			// Handle last column chunk of C
			// Need to check a condition to avoid out-of-bounds access
			column_idx = (column_stages-1)*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS + tx;
			if(column_idx < NB){
				C[I*MB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + column_idx] = __fmaf_rz(AIJ, B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][column_idx], 0.0f);
			}
		}
	}


	//
	// Adapative Kronecker Operator - invoke the corresponding Core Operator based on problem size
	//
	torch::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B){	
		validate_input(A, B);

		// Get Problem Size + Initialize C
		const problem_size_t problem_size = get_kronecker_problem_size(A, B);

		// Initialize C, threadblocks, threads
		torch::Tensor C;
		dim3 threadblocks;
		dim3 threads;

		// Batched / 3D Problem
		if(problem_size.batched){
			C = torch::empty({A.size(0), problem_size.MC, problem_size.NC}, A.options());
			threadblocks = dim3(A.size(0) * problem_size.MA, problem_size.NA, problem_size.num_stages);
			threads = dim3(SM80_KRONECKER_PROBLEM_THREADS);

			kronecker_3d_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
				A.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
				B.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
				C.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
				problem_size.MA, 
				problem_size.NA, 
				problem_size.MB, 
				problem_size.NB, 
				problem_size.MC, 
				problem_size.NC,
				(problem_size.NB + SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS - 1U) / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS
			); 

		// Unbatched / 2D Problem
		} else {
			C = torch::empty({problem_size.MC, problem_size.NC}, A.options());
			threadblocks = dim3(problem_size.MA, problem_size.NA, problem_size.num_stages);
			threads = dim3(SM80_KRONECKER_PROBLEM_THREADS);

			kronecker_2d_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
				A.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
				B.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
				C.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
				problem_size.MA, 
				problem_size.NA, 
				problem_size.MB, 
				problem_size.NB, 
				problem_size.MC, 
				problem_size.NC,
				(problem_size.NB + SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS - 1U) / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS
			); 
		}

		return C;
	}
}
