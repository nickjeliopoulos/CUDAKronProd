#include "dual_fc.cuh"
#include "../cuda_helper.cuh"


namespace winter2024::dual{
	namespace {
		constexpr int32_t SM80_WARP_SIZE = 32;
		constexpr int32_t SM80_DUAL_PROBLEM_THREADS = 256;
		constexpr int32_t SM80_DUAL_PROBLEM_WARPS = SM80_DUAL_PROBLEM_THREADS / SM80_WARP_SIZE;
		constexpr int32_t SM80_DUAL_COMPUTE_B_CHUNKS_SIZE_ROWS = 1;
		constexpr int32_t SM80_DUAL_COMPUTE_B_CHUNKS_SIZE_COLS = SM80_DUAL_PROBLEM_THREADS;
	}


	__global__ void dual_conv1x1_fp32_cuda_kernel(
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
		// Constants
		

		// Indexing
        int32_t m = blockIdx.x;
        int32_t n = blockIdx.y * blockDim.x + threadIdx.x;

		float left = b;
		float right = c;        
		
		// Allocate shared memory to store a tile of the x row.
        // extern __shared__ float x_tile[SM80_DUAL_PROBLEM_THREADS];

		// Process the input in tiles.
		for (int32_t tile_start = 0; tile_start < K; tile_start += 1) {
			left += x[m][tile_start] * W[tile_start][n];
			right += x[m][tile_start] * V[tile_start][n];
		}
		__syncthreads();

		C[m][n] = left;
		C[m][n+N] = right;	
    }


    torch::Tensor dual_fc(const torch::Tensor& x, 
                          const torch::Tensor& W, 
                          const torch::Tensor& V, 
                          const torch::Tensor& b, 
                          const torch::Tensor& c
						) {
        torch::Tensor C = torch::empty({x.size(0), 2 * W.size(1)}, x.options());
        int32_t B = x.size(0);
        int32_t D_in = x.size(1);
        int32_t D_out = W.size(1);

        dim3 threads(SM80_DUAL_PROBLEM_THREADS);
        dim3 blocks(B, (D_out + threads.x - 1) / threads.x);

        dual_conv1x1_fp32_cuda_kernel<<<blocks, threads>>>(
            x.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            W.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            V.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            C.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            b.item<float>(), 
            c.item<float>(),
            B, 
            D_in, 
            D_out
        );
        
        return C;
    }
}