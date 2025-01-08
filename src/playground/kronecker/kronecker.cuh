#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>


namespace winter2024::kronecker {
    // Core Operators
    torch::Tensor kronecker_tiny_product(const torch::Tensor& A, const torch::Tensor& B);
	torch::Tensor kronecker_anyrow_smem_product(const torch::Tensor& A, const torch::Tensor& B);
    torch::Tensor kronecker_anyrow_product(const torch::Tensor& A, const torch::Tensor& B);
    torch::Tensor kronecker_anyrow_anycol_product(const torch::Tensor& A, const torch::Tensor& B);
    // Adapative Operator which invokes Core operators based on input size
    torch::Tensor kronecker_product(const torch::Tensor& A, const torch::Tensor& B);


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "Winter 2024 - CUDA Programming Side Project";
        m.def("kronecker_tiny_product", &kronecker_tiny_product, "Kronecker Product Operator (Tiny)");
        m.def("kronecker_anyrow_smem_product", &kronecker_anyrow_smem_product, "Kronecker Product Operator (Any-Row Dim, SMEM prefetch)");
        m.def("kronecker_anyrow_product", &kronecker_anyrow_product, "Kronecker Product Operator (Any-Row Dim)");
        m.def("kronecker_anyrow_anycol_product", &kronecker_anyrow_anycol_product, "Kronecker Product Operator (Any-Row Dim, Any-Col Dim)");
        m.def("kronecker_product", &kronecker_product, "Kronecker Product Operator (Adaptive)");
    }
}