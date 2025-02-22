#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>


namespace winter2024::swiglu {
    torch::Tensor swiglu(const torch::Tensor& x, const torch::Tensor& W, const torch::Tensor& V, const torch::Tensor& b, const torch::Tensor& c);

    // Register the operators to PyTorch via PyBind11
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "Winter 2024 - CUDA Programming Side Project";
        m.def("swiglu", &swiglu, "Optimized SwiGLU");
    }
}