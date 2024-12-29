#include "kronecker.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::kronecker {
	namespace{
		using uint = unsigned int;

		struct __builtin_align__(32) problem_size_t{
			uint MA, NA;
			uint MB, NB;
			uint MC, NC;
			uint num_stages;
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
		return problem_size;
	}


	// Kernels for Kronecker Product
	// Each threadblock is responsible for a chunk of C, A[i][j] * B
	// Each thread computes a A[i][j] * B[k][l] 
	// ASSUMPTION #1: MB*NB <= SM80_KRONECKER_PROBLEM_THREADS
	// TODO: Eliminate this constraint
    template <typename scalar_t>
	__global__ void kronecker_tiny_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C,
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
		const float A_IJ = A[I][J];

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
	}


	// Kronecker Product Kernel that handles arbitrary row shapes, but assumes that at least ONE column of B fits into shared memory
	// Idea: We could also parallelize over "stages" in the kernel launch
	// Idea: We can emit code given a problem size, and generate a kernel that is optimized for that problem size
	// E.g., number of stages is constant, and we can unroll the loop, things like that
   template <typename scalar_t>
	__global__ void kronecker_anyrow_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C,
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC){
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint stage_id = blockIdx.z;
		const uint tx = threadIdx.x;

		// DEBUGGING
		assert(SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS == NB, "NB must be equal to SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS");
		static_assert(SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS == SM80_KRONECKER_PROBLEM_THREADS, "SMEM B Cols should match # of threads");

		// Prefill SMEM with chunk of B
		#pragma unroll
		for(uint k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			C[I*NB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + tx] = A[I][J] * B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][tx];
		}
	}


   template <typename scalar_t>
	__global__ void kronecker_anyrow_smem_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C,
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC){
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint stage_id = blockIdx.z;
		const uint tx = threadIdx.x;

		__shared__ float smem_B[SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS][SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS];
		const float AIJ = A[I][J];

		// DEBUGGING
		assert(SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS == NB, "NB must be equal to SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS");
		static_assert(SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS == SM80_KRONECKER_PROBLEM_THREADS, "SMEM B Cols should match # of threads");

		// Prefill SMEM with chunk of B
		#pragma unroll
		for(uint k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			// Load B into SMEM
			smem_B[k][tx] = B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][tx];
		}
		__syncthreads();

		#pragma unroll
		for(uint k = 0; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++){
			C[I*NB + stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][J*NB + tx] = AIJ * B[stage_id*SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][tx];
		}
	}


	// Operator for Kronecker Product
	torch::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B){	
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

		// DEBUGGING
		assert(problem_size.num_stages > 0, "Number of stages must be greater than 0");

		// CUDA Kernel Launch:
		// Set Kernel Launch Parameters + Launch
		const dim3 threadblocks(problem_size.MA, problem_size.NA, problem_size.num_stages);
		const dim3 threads(SM80_KRONECKER_PROBLEM_THREADS);

		// Torch Dispatch
		// TODO: Decide implementation based on problem size
		AT_DISPATCH_FLOATING_TYPES(C.type(), "gorby_kronecker", ([&]{ 
			kronecker_anyrow_smem_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
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
		}));

		return C;
	}
}
