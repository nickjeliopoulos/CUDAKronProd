#include "swiglu.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::swiglu{

    __device__ float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    __device__ float swish(float x) {
        return x * sigmoid(x);
    }

    // Tiled version: each block loads a tile of the input row (x[batch]) into shared memory.
    template <typename scalar_t>
    __global__ void swiglu_2d_fp32_cuda_kernel(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> W, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> V, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C,
        scalar_t b,
        scalar_t c,
        int B, 
        int D_in, 
        int D_out
    ) {
        // Each block handles one batch
        int batch = blockIdx.x;
        // Each thread computes one output dimension.
        int d_out = blockIdx.y * blockDim.x + threadIdx.x;

        // Allocate shared memory to store a tile of the x row.
        extern __shared__ scalar_t x_tile[];

        if (batch < B && d_out < D_out) {
            scalar_t left = b;
            scalar_t right = c;
            
            // Use the block dimension as the tile size.
            int tile_size = blockDim.x;

            // Process the input in tiles.
            for (int tile_start = 0; tile_start < D_in; tile_start += tile_size) {
                // Each thread loads one element of the tile.
                int idx = tile_start + threadIdx.x;
                if (idx < D_in) {
                    x_tile[threadIdx.x] = x[batch][idx];
                } else {
                    x_tile[threadIdx.x] = 0;
                }
                __syncthreads();
                
                // Calculate the number of valid elements in this tile.
                int current_tile_size = (D_in - tile_start < tile_size) ? (D_in - tile_start) : tile_size;

                // Iterate over the tile elements and update partial sums.
                for (int i = 0; i < current_tile_size; i++) {
                    scalar_t x_val = x_tile[i];
                    left += x_val * W[tile_start + i][d_out];
                    right += x_val * V[tile_start + i][d_out];
                }
                __syncthreads();

            }
			
            C[batch][d_out] = swish(left) * right;
        }
    }

    torch::Tensor swiglu(const torch::Tensor& x, 
                          const torch::Tensor& W, 
                          const torch::Tensor& V, 
                          const torch::Tensor& b, 
                          const torch::Tensor& c) {
        auto C = torch::empty({x.size(0), W.size(1)}, x.options());
        int B = x.size(0);
        int D_in = x.size(1);
        int D_out = W.size(1);

        dim3 threads(256);
        dim3 blocks(B, (D_out + threads.x - 1) / threads.x);
        // Allocate shared memory equal to one tile (the size of one row segment of x)
        size_t sharedMemSize = threads.x * sizeof(float);

        swiglu_2d_fp32_cuda_kernel<float><<<blocks, threads, sharedMemSize>>>(
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
