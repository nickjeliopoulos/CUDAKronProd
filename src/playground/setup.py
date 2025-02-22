import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, COMMON_MSVC_FLAGS


if os.name == "nt":
    COMMON_MSVC_FLAGS += ['/Zc:__cplusplus']


setup(
    ### Minimal Example Args
    name="gorby-kronecker",
    install_requires=["torch >= 2.1", "pybind11"],
    ### PyTorch C++/CUDA Examples
    ### NOTE: /Zc:__cplusplus is related to MSVC incorrectly setting __cplusplus macro. See CUTLASS CMAKE and 
    ### https://github.com/NVIDIA/cutlass/issues/1474
    ext_modules=[
        # CUDAExtension(
        #    name="gorby_kronecker", sources=["kronecker/kronecker.cu"],
        # ),
        CUDAExtension(
            name="gorby_swiglu", sources=["swiglu/swiglu.cu"], extra_compile_args={'nvcc' : ["-arch=sm_80"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)