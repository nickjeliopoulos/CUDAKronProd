<div align="center">

# Winter Break 2024: From-Scratch Kronecker Product for PyTorch
</div>

<div align="center">

This repo contains a CUDA/C++ kernel that implements the Kronecker Product, which I wrote from scratch over the winter break - primarily as a practical exercise in kernel development.

</div>

<!-- Installation Guide -->
# Installation 
```bash
cd src/playground/
python setup.py develop
```

<!-- Usage Guide -->
# Usage 
> [!CAUTION]
> To best reproduce latency measurements, we encourage users to lock the clock and/or memory rates of their device.

## benchmark.py
This is a (placeholder) example that computes the Kronecker product between two matrices, A and B, each with *M* rows and *N* columns.
```bash
cd src/playground/
python benchmark.py --M 32 --N 512
> Reference median latency: 14.2 ms
> My median latency: 12.8 ms
> Numerically Close? True
```

# Insights and Post-Mortem Thoughts
> [!NOTE]
> See `src/playground/README.md` for a more in-depth write-up of profiling results.
