#include "dual_fc.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::dual{
    namespace {
        constexpr int32_t SM80_WARP_SIZE = 32;
        constexpr int32_t SM80_DUAL_PROBLEM_THREADS = 256;
        constexpr int32_t SM80_DUAL_PROBLEM_WARPS = SM80_DUAL_PROBLEM_THREADS / SM80_WARP_SIZE;
    }

    // This kernel fuses the two matrix multiplications:
    //   left = xW + b and right = xV + c
    // and writes the results to C such that:
    //   C[batch][0:D_out] = left, C[batch][D_out:2*D_out] = right.
    //
    // The kernel uses shared memory to load a tile of x for each batch,
    // reducing global memory traffic by reusing the same x tile for both multiplications.
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
		// Constants
		constexpr int32_t tile_size = SM80_DUAL_PROBLEM_THREADS;

        // Thread Indexing
        int32_t batch = blockIdx.x;
        int32_t d_out = blockIdx.y * SM80_DUAL_PROBLEM_THREADS + threadIdx.x;

		// Thread-local registers
		float left = b;
		float right = c;

		// SMEM
        // extern __shared__ float smem[3][4096];

        // Only proceed if within valid bounds.
        if (d_out < D_out) {
			// Blocking the D_in dimensions (GEMM K dimension)
			for (int32_t din_offset = 0; din_offset < D_in; din_offset += tile_size) {
				// SMEM Loading
                // if (idx < D_in) {
                //  smem[0][idx] = x[batch][din_idx];
				// 	smem[1][idx] = W[din_idx][d_out];
				// 	smem[2][idx] = V[din_idx][d_out];
                // } else {
                //     smem[0][idx] = 0.0f;
				// 	smem[1][idx] = 0.0f;
				// 	smem[2][idx] = 0.0f;                
				// }
                // __syncthreads();

                // Determine how many elements are valid in this tile.
                int32_t current_tile_size = (D_in - din_offset < tile_size) ? (D_in - din_offset) : tile_size;

                // Each thread iterates over the current tile to accumulate partial results.
                // The same tile is used for both the W and V multiplications.
                for (int32_t din_idx = 0; din_idx < current_tile_size; din_idx++) {
					int32_t idx = din_offset + din_idx;
					// Speedy SMEM Access
                    // left += smem[0][i] * smem[1][i];
                    // right += smem[0][i] * smem[2][i];

					// Slow Global Memory Access
					left += x[batch][idx] * W[idx][d_out];
					right += x[batch][idx] * V[idx][d_out];
                }
                __syncthreads();
            }

            // Write the fused results:
            // First half of C holds (xW + b), and the second half holds (xV + c).
            C[batch][d_out] = left;
            C[batch][D_out + d_out] = right;
        }
    }

    // Host API for the dual_fc operation.
    torch::Tensor dual_fc(const torch::Tensor& x, 
                          const torch::Tensor& W, 
                          const torch::Tensor& V, 
                          const torch::Tensor& b, 
                          const torch::Tensor& c
    ) {
        // Allocate output tensor with shape (B, 2*D_out)
        torch::Tensor C = torch::empty({x.size(0), 2 * W.size(1)}, x.options());
        int32_t B = x.size(0);
        int32_t D_in = x.size(1);
        int32_t D_out = W.size(1);

        // Configure a 2D grid: one block per batch, and blocks along the second dimension to cover D_out.
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
