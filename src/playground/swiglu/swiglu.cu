#include "swiglu.cuh"
#include "../cuda_helper.cuh"

namespace winter2024::swiglu{

    __device__ float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    __device__ float swish(float x) {
        return x * sigmoid(x);
    }

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
        int batch = blockIdx.x;  // Each block handles one batch
        int d_out = blockIdx.y * blockDim.x + threadIdx.x; // Each thread handles one output dimension

		scalar_t left = b;
		scalar_t right = c;
		scalar_t x_val = 0;

		if (batch < B && d_out < D_out) {
			for (int d_in = 0; d_in < D_in; d_in++) {
				x_val = x[batch][d_in];
				left += x_val * W[d_in][d_out];
				right += x_val * V[d_in][d_out];
			}

			C[batch][d_out] = swish(left) * right;
		}
    }

    torch::Tensor swiglu(const torch::Tensor& x, const torch::Tensor& W, const torch::Tensor& V, const torch::Tensor& b, const torch::Tensor& c) {
        auto C = torch::empty({x.size(0), W.size(1)}, x.options());
        int B = x.size(0);
        int D_in = x.size(1);
        int D_out = W.size(1);

        dim3 threads(256);
        dim3 blocks(B, (D_out + threads.x - 1) / threads.x);
		size_t sharedMemSize = sizeof(float) * threads.x * 2;

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
