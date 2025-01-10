<div align="center">

# Profiling Insights 
Here, we perform an analysis of baseline Kronecker product kernel performance via PyTorch's `torch.kron(...)` and my implementation.

</div>

# Benchmarking Procedure
Kernel performance measurements (aside from latency) were performed using [NVIDIA NSight Compute](https://developer.nvidia.com/nsight-compute). All measurementss were performed on a desktop equipped with a NVIDIA RTX 3090Ti GPU.
Latency and numerical correctness were measured in the `correctness.py` script separately.

## Latency
In order to best approximate the latency for kernel invocations, we report the median latency over 128 runs for `troch.kron(...)` and my implementation. **We do this separately because NSight Compute serializes GPU computations in order to profile them, which would greatly skew latency.**

## Numerical Correctness
We use the PyTorch function `torch.allclose(...)` to report whether the resulting Kronecker product from our kernel matches that of `torch.kron(...)`.

## Workload Sizes
In order to assess the kernel performance in relation to PyTorch's `torch.kron(...)`, I profiled across a wide range of input shapes, illustrated below (the full set can be seen in `correctness.py`).
These workloads span small, "toy" workload sizes up to typical deep learning matrix sizes that involve Kronecker Products (see REFERENCE XXX, YYY).

# Analysis

# Summary

