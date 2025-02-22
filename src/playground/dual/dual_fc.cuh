#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>


namespace winter2024::dual {
	/**
	 * @file dual_fc.cuh
	 * @brief Performs dual convolution within a single Tensor.
	 *
	 * This file contains the implementation of the dual convolution operation.
	 *
	 * @param x Input tensor of shape (B, D_in).
	 * @param W Weight tensor of shape (D_in, D_out).
	 * @param V Weight tensor of shape (D_in, D_out).
	 * @param b Scalar bias term.
	 * @param c Scalar bias term.
	 * @return Result of the dual convolution as a tensor of shape (B, 2 * D_out). 
	 */
    torch::Tensor dual_fc(
		const torch::Tensor& x, 
		const torch::Tensor& W, 
		const torch::Tensor& V, 
		const torch::Tensor& b, 
		const torch::Tensor& c
	);

    // Register the operators to PyTorch via PyBind11
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "Winter 2024 - CUDA Programming Side Project";
        m.def("dual_fc", &dual_fc, "Dual 1x1 Convolution / Linear Layer");
    }
}