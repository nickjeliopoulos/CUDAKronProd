#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>


namespace winter2024::kronecker {
	torch::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B);


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "Winter 2024 - CUDA Programming Side Project";
        m.def("kronecker_product", &kronecker_product, "Kronecker Product Operator");
    }
}