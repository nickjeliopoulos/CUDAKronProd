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

		struct __builtin_align__(8) dim2{
			uint x, y;
			dim2(uint x, uint y) : x(x), y(y) {}
		};

		constexpr uint SM80_THREADS_PER_WARP = 32;
		constexpr uint SM80_MAX_THREADS = 1536;
		constexpr uint SM80_MAX_WARPS = SM80_MAX_THREADS / SM80_THREADS_PER_WARP;
		constexpr uint SM80_KRONECKER_PROBLEM_THREADS = 256;
		constexpr uint SM80_KRONECKER_PROBLEM_WARPS = SM80_KRONECKER_PROBLEM_THREADS / SM80_THREADS_PER_WARP;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS = 4;
		constexpr uint SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS = 256;
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
		return problem_size;
	}


	// Kernels for Kronecker Product
	// Each threadblock is responsible for a chunk of C, A[i][j] * B
	// Each thread computes a A[i][j] * B[k][l] 
	// ASSUMPTION #1: MB*NB <= SM80_KRONECKER_PROBLEM_THREADS
	// TODO: Eliminate this constraint
    template <typename scalar_t>
	__global__ void kronecker_tiny_sm80_fp32_fp32_cuda_kernel(
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
		__shared__ float smem_B[64][64];
		__shared__ float A_IJ = A[I][J];

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


	// Inner function that prefills Shared Memory with chunks of B (start of each stage)
	// Current constraints:
	// 1) 1+ rows of B will fit into smem. A threadblock is responsible for operating on a a chunk of B at a time (what is in smem).
	template <typename scalar_t>
	__device__ __inline_hint__ void _inner_prefill_kronecker_anysize_sm80_fp32_fp32_cuda_kernel(
		// Pointers
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B, 
		float smem_B[SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS][SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS],
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC,
		// Thread ID
		const uint stage,
		const uint I,
		const uint J,
		const uint tx){
		// Load elements of B into Shared Memory
		// Identify how many elements of B to prefill each thread is responsible for
		uint num_elements_per_thread = min(1, ( SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS * SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS ) / ( SM80_KRONECKER_PROBLEM_THREADS ) );
		uint thread_prefill_idx = 0;
		uint thread_element_idx = tx * num_elements_per_thread;
		uint k = 0;
		uint l = 0;

		// Note: If SM80_KRONECKER_PROBLEM_THREADS > NB, there may be a situation where we are out of bounds
		// As a result, we would need to adjust num_elements_per_thread
		if ( thread_element_idx >= NB ){
			// Unsure if min(0, ...) is necessary. Need to investigate if this could happen - but it seems "safer" to have it
			num_elements_per_thread = min(0, NB - ( tx * num_elements_per_thread ))
		}
		// Unsure if necessary - but here we don't necessarily need a __synthread(...), just a syncwarp.
		// Why? All threads / warps before the last warp should be responsible for a dense chunk of B that does not go out of bounds
		// For that reason I think a __syncthreads(...) is unnecessary, because the rest of the threadblock should have completed that work
		// Depending on problem size and shared memory configuration, one may be able to guarantee that we will not go out of bounds and the above
		// check is unncesarry, along with this syncwarp. Need to investigate
		__syncwarp(SYNCWARP_ALL_MASK);

		// Each thread will load "num_elements_per_thread" elements of B into smem
		for( ; thread_prefill_idx < num_elements_per_thread; thread_prefill_idx++ ){
			k = thread_element_idx / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS;
			l = thread_element_idx % SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS;
			smem_B[k][l] = B[stage * SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS + k][l];
			// Increment thread element index
			thread_element_idx++;
		}
	}


	// Kronecker Product Kernel that handles arbitrary tensor dimensions
	// This kernel uses shared memory, and operates in stages
	// Chunk parts of B into SMEM at a time, compute, then repeat (next stage)
	// TODO: Finish Up, should be better then "naive" implementation
   template <typename scalar_t>
	__global__ void kronecker_anysize_sm80_fp32_fp32_cuda_kernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B, 
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
		// Problem Size
		const uint MA, 
		const uint NA, 
		const uint MB, 
		const uint NB, 
		const uint MC, 
		const uint NC){
		// Thread ID
		const uint I = blockIdx.x;
		const uint J = blockIdx.y;
		const uint tx = threadIdx.x;

		// Shared Memory and Variable Init.
		// TODO: Compute Shared Memory Size based on problem size, data type etc (shared memory size per SM is limited, based on # of threadblocks etc.)
		__shared__ float A_IJ = A[I][J];
		__shared__ float smem_B[SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS][SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS];

		// ASSUMPTION: MB / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS >= 1
		// If problem size is small enough, consider using naive implementation
		const uint num_stages = min(1, MB / SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS);
		uint stage = 0;

		for ( ; stage < num_stages; stage++ ){
			// Prefill Shared Memory with chunk of B
			_inner_prefill_kronecker_anysize_sm80_fp32_fp32_cuda_kernel(B, smem_B, MA, NA, MB, NB, MC, NC, stage, I, J, tx);
			__syncthreads();

			// Compute chunk of C
			uint k = 0;
			uint l = 0;

			// Compute bounds for loops

			for( ; k < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_ROWS; k++ ){
				for( ; l < SM80_KRONECKER_COMPUTE_B_CHUNKS_SIZE_COLS; l++ ){
					C[I*MB + k][J*NB + l] = A_IJ * smem_B[k][l];
				}
			}
			__syncthreads();
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

		// CUDA Kernel Launch:
		// Set Kernel Launch Parameters + Launch
		const dim3 threadblocks(problem_size.MA, problem_size.NA);
		const dim3 threads(SM80_KRONECKER_PROBLEM_THREADS);

		// Torch Dispatch
		// TODO: Decide implementation based on problem size
		AT_DISPATCH_FLOATING_TYPES(C.type(), "gorby_kronecker", ([&]{ 
			kronecker_tiny_sm80_fp32_fp32_cuda_kernel<float><<<threadblocks, threads>>>(
				A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
				B.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
				C.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
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
