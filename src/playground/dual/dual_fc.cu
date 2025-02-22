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

	
    // This kernel performs the fused matrix multiplications:
    //   left = xW + b and right = xV + c,
    // writing them to C such that:
    //   C[batch][0:D_out] = left and C[batch][D_out:2*D_out] = right.
    //
    // In addition to caching the input vector x in shared memory,
    // we now also cache the corresponding tile elements of W and V.
    __global__ void dual_conv1x1_fp32_cuda_kernel(
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> x, 
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> W, 
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> V, 
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> C,
        float b,
        float c,
        int32_t B, 
        int32_t D_in, 
        int32_t D_out
    ) {
        // Each block handles one batch.
        int32_t batch = blockIdx.x;
        // Each thread computes one output column.
        int32_t d_out = blockIdx.y * blockDim.x + threadIdx.x;

        // Use shared memory to cache tiles from x, W, and V.
        // Partition smem into three contiguous arrays:
        //   - x_tile: holds a tile of x (size: blockDim.x)
        //   - W_tile: holds the corresponding tile of W (for the given d_out) (size: blockDim.x)
        //   - V_tile: holds the corresponding tile of V (for the given d_out) (size: blockDim.x)
        extern __shared__ float smem[];
        float* x_tile = smem;                          // size: blockDim.x
        float* W_tile = x_tile + blockDim.x;           // size: blockDim.x
        float* V_tile = W_tile + blockDim.x;           // size: blockDim.x

        if (batch < B && d_out < D_out) {
            float left = b;    // Accumulator for xW + b.
            float right = c;   // Accumulator for xV + c.
            int32_t tile_size = blockDim.x;

            // Process the input dimension in tiles.
            for (int32_t tile_start = 0; tile_start < D_in; tile_start += tile_size) {
                int32_t idx = tile_start + threadIdx.x;
                // Load a tile of x.
                if (idx < D_in)
                    x_tile[threadIdx.x] = x[batch][idx];
                else
                    x_tile[threadIdx.x] = 0.0f;
                
                // Load the corresponding element from W and V.
                if (idx < D_in) {
                    W_tile[threadIdx.x] = W[tile_start + threadIdx.x][d_out];
                    V_tile[threadIdx.x] = V[tile_start + threadIdx.x][d_out];
                } else {
                    W_tile[threadIdx.x] = 0.0f;
                    V_tile[threadIdx.x] = 0.0f;
                }
                __syncthreads();

                int32_t current_tile_size = (D_in - tile_start < tile_size) ? (D_in - tile_start) : tile_size;
                // Use the cached tiles to update the accumulators.
                for (int32_t i = 0; i < current_tile_size; i++) {
                    float x_val = x_tile[i];
                    left  += x_val * W_tile[i];
                    right += x_val * V_tile[i];
                }
                __syncthreads();
            }

            // Write the results:
            // First half of C holds left (xW + b) and the second half holds right (xV + c).
            C[batch][d_out] = left;
            C[batch][D_out + d_out] = right;
        }
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
        // Allocate shared memory equal to one tile (the size of one row segment of x)
        size_t sharedMemSize = SM80_DUAL_PROBLEM_THREADS * sizeof(float) * 3;

        dual_conv1x1_fp32_cuda_kernel<<<blocks, threads, sharedMemSize>>>(
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
