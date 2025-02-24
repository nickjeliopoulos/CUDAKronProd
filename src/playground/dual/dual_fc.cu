#include "dual_fc.cuh"
#include "../cuda_helper.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace winter2024::dual{
	namespace {
		constexpr int32_t SM86_DUAL_PROBLEM_THREADS = 256;

		// Constants
		constexpr int32_t BLOCK_M = 16;
		constexpr int32_t BLOCK_N = 16;
		constexpr int32_t BLOCK_K = 8;
		constexpr int32_t BLOCK_RESULTS = BLOCK_M * BLOCK_N;
		constexpr int32_t THREAD_TM = 2;
		constexpr int32_t THREAD_TN = 2;
		constexpr int32_t THREAD_RESULTS = THREAD_TM * THREAD_TN;
	}


	__global__ void dual_fc_fp32_cuda_kernel(
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> x, 
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> W, 
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> V, 
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> C,
        float b,
        float c,
        int32_t M, 
        int32_t K, 
        int32_t N
    ) {
		// Indexing
		const int32_t c_row = blockIdx.y;
		const int32_t c_col = blockIdx.x;

		const int32_t thread_row = threadIdx.x / (BLOCK_N / THREAD_TN);
		const int32_t thread_col = threadIdx.x % (BLOCK_N / THREAD_TN);

		const int32_t inner_row_x_stride = blockDim.x / BLOCK_K;
		const int32_t inner_row_x = threadIdx.x / BLOCK_K;
		const int32_t inner_col_x = threadIdx.x % BLOCK_K;

		const int32_t inner_row_WV_stride = blockDim.x / BLOCK_N;
		const int32_t inner_row_WV = threadIdx.x / BLOCK_N;
		const int32_t inner_col_WV = threadIdx.x % BLOCK_N;

		// Block offsets
		const int32_t threadblock_x_row_offset = c_row * BLOCK_M;
		const int32_t threadblock_wv_col_offset = c_col * BLOCK_N;
		const int32_t threadblock_c_row_offset = threadblock_x_row_offset;
		const int32_t threadblock_c_col_offset = threadblock_wv_col_offset;

		// Init Shared Memory
		__shared__ float shared_x[BLOCK_M][BLOCK_K];
		__shared__ float shared_W[BLOCK_K][BLOCK_N];
		__shared__ float shared_V[BLOCK_K][BLOCK_N];

		// Thread-local intermediate result
		float thread_result_xW_cache[THREAD_TM][THREAD_TN] = {0.0f};
		float thread_result_xV_cache[THREAD_TM][THREAD_TN] = {0.0f};
		float thread_local_x_cache[THREAD_TM] = {0.0f};
		float thread_local_W_cache[THREAD_TN] = {0.0f};
		float thread_local_V_cache[THREAD_TN] = {0.0f};
		
		// Question. Is it better to do W, V related operations at the same time, or is it better do do xW + b, then xV + c?
		// Theoretically this might help with live registers / register pressure if they were done sequentially.
		// Unsure about warp scheduling effects however.
		// Outer Block Loop
		for( int32_t block_k = 0; block_k < K; block_k += BLOCK_K ) {
			// SMEM x Population
			for( int32_t m_offset = 0; m_offset < BLOCK_M; m_offset += inner_row_x_stride ) {
				shared_x[m_offset + inner_row_x][inner_col_x] = x[m_offset + inner_row_x + threadblock_x_row_offset][inner_col_x + block_k];
			}
			// SMEM W,V Population
			for( int32_t k_offset = 0; k_offset < BLOCK_K; k_offset += inner_row_WV_stride ) {
				shared_W[k_offset + inner_row_WV][inner_col_WV] = W[k_offset + inner_row_WV + block_k][inner_col_WV + threadblock_wv_col_offset];
				shared_V[k_offset + inner_row_WV][inner_col_WV] = V[k_offset + inner_row_WV + block_k][inner_col_WV + threadblock_wv_col_offset];
			}
			__syncthreads();
			
			// Inner Thread Compute Loop
			for (int32_t thread_block_k = 0; thread_block_k < BLOCK_K; ++thread_block_k) {
				#pragma unroll
				for(int32_t thread_tm_idx = 0; thread_tm_idx < THREAD_TM; ++thread_tm_idx) {
					thread_local_x_cache[thread_tm_idx] = shared_x[thread_row*THREAD_TM + thread_tm_idx][thread_block_k];
				}
				#pragma unroll
				for(int32_t thread_tn_idx = 0; thread_tn_idx < THREAD_TN; ++thread_tn_idx) {
					thread_local_W_cache[thread_tn_idx] = shared_W[thread_block_k][thread_col*THREAD_TN + thread_tn_idx];
					thread_local_V_cache[thread_tn_idx] = shared_V[thread_block_k][thread_col*THREAD_TN + thread_tn_idx];
				}
				#pragma unroll
				for(int32_t thread_tm_compute_idx = 0; thread_tm_compute_idx < THREAD_TM; ++thread_tm_compute_idx) {
					for(int32_t thread_tn_compute_idx = 0; thread_tn_compute_idx < THREAD_TN; ++thread_tn_compute_idx) {
						thread_result_xW_cache[thread_tm_compute_idx][thread_tn_compute_idx] += thread_local_x_cache[thread_tm_compute_idx] * thread_local_W_cache[thread_tn_compute_idx];
						thread_result_xV_cache[thread_tm_compute_idx][thread_tn_compute_idx] += thread_local_x_cache[thread_tm_compute_idx] * thread_local_V_cache[thread_tn_compute_idx];
					}
				}
			}
			__syncthreads();
		}

		// Store Results
		#pragma unroll
		for(int32_t thread_tm_idx = 0; thread_tm_idx < THREAD_TM; ++thread_tm_idx) {
			for(int32_t thread_tn_idx = 0; thread_tn_idx < THREAD_TN; ++thread_tn_idx) {
				C[thread_row * THREAD_TM + thread_tm_idx + threadblock_c_row_offset][thread_col * THREAD_TN + thread_tn_idx + threadblock_c_col_offset    ] = thread_result_xW_cache[thread_tm_idx][thread_tn_idx] + b;	
				C[thread_row * THREAD_TM + thread_tm_idx + threadblock_c_row_offset][thread_col * THREAD_TN + thread_tn_idx + threadblock_c_col_offset + N] = thread_result_xV_cache[thread_tm_idx][thread_tn_idx] + c;		
			}
		}
    }


    torch::Tensor dual_fc(const torch::Tensor& x, 
                          const torch::Tensor& W, 
                          const torch::Tensor& V, 
                          const torch::Tensor& b, 
                          const torch::Tensor& c
						) {
        torch::Tensor C = torch::empty({x.size(0), 2 * W.size(1)}, x.options());
        int32_t M = x.size(0);
        int32_t K = x.size(1);
        int32_t N = W.size(1);

		dim3 grid(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M), 1);
		dim3 threads(BLOCK_RESULTS / THREAD_RESULTS, 1, 1);

        dual_fc_fp32_cuda_kernel<<<grid, threads>>>(
            x.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            W.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            V.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            C.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            b.item<float>(), 
            c.item<float>(),
            M, 
            K, 
            N
        );
        
        return C;
    }
}