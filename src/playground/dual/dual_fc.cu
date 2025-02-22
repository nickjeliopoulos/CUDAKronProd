#include "dual_fc.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::dual{
    namespace {
        constexpr int32_t SM80_WARP_SIZE = 32;
        constexpr int32_t SM80_DUAL_PROBLEM_THREADS = 256;
        // Number of warps per block:
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
        // Each block handles one batch.
        int32_t batch = blockIdx.x;
        // Each thread computes one output dimension.
        int32_t d_out = blockIdx.y * blockDim.x + threadIdx.x;

        // Declare shared memory to hold a tile of the x row.
        // The tile size is chosen to be blockDim.x.
        extern __shared__ float x_tile[];

        // Only proceed if within valid bounds.
        if (batch < B && d_out < D_out) {
            float left = b;   // Accumulator for xW + b.
            float right = c;  // Accumulator for xV + c.

            // Define the tile size as the number of threads.
            int32_t tile_size = blockDim.x;

            // Process the input vector x[batch] in tiles.
            for (int32_t tile_start = 0; tile_start < D_in; tile_start += tile_size) {
                // Each thread loads one element of x into shared memory.
                int32_t idx = tile_start + threadIdx.x;
                if (idx < D_in) {
                    x_tile[threadIdx.x] = x[batch][idx];
                } else {
                    // For out-of-bound threads in this tile, load zero.
                    x_tile[threadIdx.x] = 0.0f;
                }
                __syncthreads();

                // Determine how many elements are valid in this tile.
                int32_t current_tile_size = (D_in - tile_start < tile_size) ? (D_in - tile_start) : tile_size;

                // Each thread iterates over the current tile to accumulate partial results.
                // The same tile is used for both the W and V multiplications.
                for (int32_t i = 0; i < current_tile_size; i++) {
                    float x_val = x_tile[i];
                    left  += x_val * W[tile_start + i][d_out];
                    right += x_val * V[tile_start + i][d_out];
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
        // Allocate shared memory for one tile of x (tile size equals number of threads).
        size_t sharedMemSize = SM80_DUAL_PROBLEM_THREADS * sizeof(float);

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
