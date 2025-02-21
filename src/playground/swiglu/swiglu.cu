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
        const scalar_t* __restrict__ A,
        const scalar_t* __restrict__ B,
        scalar_t* __restrict__ C,
        int numel
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            C[idx] = swish(A[idx]) * B[idx];
        }
    }

    torch::Tensor swiglu(const torch::Tensor& A, const torch::Tensor& B) {
        auto C = torch::empty_like(A);
        int numel = A.numel();
        int threads = 256;
        int blocks = (numel + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "swiglu_cuda", ([&] {
            swiglu_2d_fp32_cuda_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), numel);
        }));

        return C;
    }
}
