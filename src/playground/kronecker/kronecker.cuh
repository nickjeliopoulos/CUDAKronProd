#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>


namespace winter2024::kronecker {
	torch::Tensor kronecker_smem_anyrow_product(const torch::Tensor& A, const torch::Tensor& B);
    torch::Tensor kronecker_anyrow_product(const torch::Tensor& A, const torch::Tensor& B);


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "Winter 2024 - CUDA Programming Side Project";
        m.def("kronecker_smem_anyrow_product", &kronecker_smem_anyrow_product, "Kronecker Product Operator (SMEM, Any-Row Dim)");
        m.def("kronecker_anyrow_product", &kronecker_anyrow_product, "Kronecker Product Operator (Any-Row Dim)");

    }
}